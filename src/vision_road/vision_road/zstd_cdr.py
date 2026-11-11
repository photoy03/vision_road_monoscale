"""Lossless local cache of original CDR bytes, using the system Zstandard library."""
import ctypes as ct
import ctypes.util
import hashlib,json
from pathlib import Path


class Codec:
    def __init__(self,capacity):
        self.lib=ct.CDLL(ct.util.find_library('zstd') or 'libzstd.so.1')
        self.lib.ZSTD_compressBound.argtypes=[ct.c_size_t];self.lib.ZSTD_compressBound.restype=ct.c_size_t
        self.lib.ZSTD_compress.argtypes=[ct.c_void_p,ct.c_size_t,ct.c_void_p,ct.c_size_t,ct.c_int];self.lib.ZSTD_compress.restype=ct.c_size_t
        self.lib.ZSTD_decompress.argtypes=[ct.c_void_p,ct.c_size_t,ct.c_void_p,ct.c_size_t];self.lib.ZSTD_decompress.restype=ct.c_size_t
        self.lib.ZSTD_isError.argtypes=[ct.c_size_t];self.lib.ZSTD_isError.restype=ct.c_uint
        self.lib.ZSTD_getErrorName.argtypes=[ct.c_size_t];self.lib.ZSTD_getErrorName.restype=ct.c_char_p
        self.capacity=capacity;self.compressed=ct.create_string_buffer(self.lib.ZSTD_compressBound(capacity));self.raw=ct.create_string_buffer(capacity)
    def check(self,size):
        if self.lib.ZSTD_isError(size):raise ValueError(self.lib.ZSTD_getErrorName(size).decode())
        return size
    def compress(self,data):
        if len(data)>self.capacity:raise ValueError('CDR exceeds cache capacity')
        size=self.check(self.lib.ZSTD_compress(self.compressed,len(self.compressed),ct.c_char_p(data),len(data),1))
        return ct.string_at(self.compressed,size)
    def decompress(self,data,size):
        result=self.check(self.lib.ZSTD_decompress(self.raw,self.capacity,ct.c_char_p(data),len(data)))
        if result!=size:raise ValueError('CDR cache size mismatch')
        return ct.string_at(self.raw,result)


def cache_records(path,db,start=0,count=0):
    path=Path(path);db=Path(db);manifest=json.loads(path.with_suffix(path.suffix+'.json').read_text())
    if manifest['version']!=1 or manifest['source_size']!=db.stat().st_size or manifest['source_mtime_ns']!=db.stat().st_mtime_ns:raise ValueError('Cache does not match the source bag')
    codec=Codec(manifest['max_raw_size']);entries=manifest['entries'][start:start+count if count else None]
    with path.open('rb') as file:
        if file.read(8)!=b'VRCDRZ01':raise ValueError('Invalid CDR cache')
        for record in entries:
            file.seek(record['offset']);data=file.read(record['compressed_size']);raw=codec.decompress(data,record['raw_size'])
            if hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('CDR cache checksum mismatch')
            yield raw
