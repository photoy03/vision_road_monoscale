import hashlib,json
import pytest
from vision_road.zstd_cdr import Codec,cache_records


def fixture_cache(tmp_path):
    raw=bytes(range(256))*64
    codec=Codec(len(raw));compressed=codec.compress(raw)
    assert codec.decompress(compressed,len(raw))==raw
    database=tmp_path/'source.db3';database.write_bytes(b'reference-only')
    cache=tmp_path/'frames.cdrz';cache.write_bytes(b'VRCDRZ01'+compressed)
    manifest=dict(version=1,source_size=database.stat().st_size,source_mtime_ns=database.stat().st_mtime_ns,max_raw_size=len(raw),entries=[dict(offset=8,compressed_size=len(compressed),raw_size=len(raw),sha256=hashlib.sha256(raw).hexdigest())])
    cache.with_suffix('.cdrz.json').write_text(json.dumps(manifest))
    return cache,database,raw,manifest


def test_cdr_cache_preserves_embedded_null_bytes(tmp_path):
    cache,database,raw,_=fixture_cache(tmp_path)
    assert list(cache_records(cache,database))==[raw]


def test_cdr_cache_rejects_wrong_checksum(tmp_path):
    cache,database,_,manifest=fixture_cache(tmp_path)
    manifest['entries'][0]['sha256']='0'*64
    cache.with_suffix('.cdrz.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='checksum'):list(cache_records(cache,database))


def test_cdr_cache_rejects_changed_source(tmp_path):
    cache,database,_,_=fixture_cache(tmp_path)
    database.write_bytes(b'changed-source-size')
    with pytest.raises(ValueError,match='source bag'):list(cache_records(cache,database))
