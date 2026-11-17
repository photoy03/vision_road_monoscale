#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
namespace py=pybind11;
using Bytes=py::array_t<unsigned char,py::array::c_style|py::array::forcecast>;
using Points=py::array_t<double,py::array::c_style|py::array::forcecast>;
struct Fragment {std::vector<int>x,y;std::vector<double>points;double m,b,lo,hi,width;bool yellow;};
py::list extract(Bytes mask,Bytes yellow,double mpp,double forward,double lateral,double reach){
 auto mb=mask.request(),yb=yellow.request();
 if(mb.ndim!=2||yb.ndim!=2||mb.shape!=yb.shape||mpp<=0)throw std::invalid_argument("Invalid fragment images");
 int h=mb.shape[0],w=mb.shape[1],tile=std::max(8,int(std::nearbyint(2./mpp)));
 auto raw=static_cast<unsigned char*>(mb.ptr);auto color=static_cast<unsigned char*>(yb.ptr);std::vector<Fragment> out;
 {
 py::gil_scoped_release release;
 cv::Mat image(h,w,CV_8U,raw);
 for(int start=0;start<h;start+=tile){
  int height=std::min(tile,h-start);cv::Mat labels,stats,centers;
  int count=cv::connectedComponentsWithStats(image.rowRange(start,start+height),labels,stats,centers,8,CV_32S);
  std::vector<std::vector<int>> xx(count),yy(count);std::vector<bool>eligible(count,false);
  for(int label=1;label<count;++label)eligible[label]=stats.at<int>(label,cv::CC_STAT_AREA)>=6&&(stats.at<int>(label,cv::CC_STAT_HEIGHT)-1)*mpp>=.65;
  for(int y=0;y<height;++y){auto row=labels.ptr<int>(y);for(int x=0;x<w;++x){int label=row[x];if(label&&eligible[label]){xx[label].push_back(x);yy[label].push_back(y+start);}}}
  for(int label=1;label<count;++label){
   if(!eligible[label])continue;auto&x=xx[label];auto&y=yy[label];int n=x.size();double mx=0,my=0;
   for(int k=0;k<n;++k){mx+=x[k];my+=y[k];}mx/=n;my/=n;double vx=0,vy=0,cov=0;int yc=0;
   for(int k=0;k<n;++k){double dx=x[k]-mx,dy=y[k]-my;vx+=dx*dx;vy+=dy*dy;cov+=dx*dy;yc+=color[y[k]*w+x[k]]!=0;}
   vx/=n-1;vy/=n-1;cov/=n-1;double disc=std::hypot(vx-vy,2*cov),axis=disc?.5*(1-(vx-vy)/disc):1.;double normal=std::sqrt(std::max((vx+vy-disc)*.5,0.))*mpp;
   if(axis<.35*.35||normal>.10||vy<=0)continue;
   double m=cov/vy,b=mx-m*my,anchor=lateral+(m*((forward-3.)/mpp)+b)*mpp;
   // A straight extrapolation is only meaningful near the car; far curved
   // fragments get their side from the chain they join.  reach > 4.2 also keeps
   // near fragments up to that far out (a neighbouring lane's outer line).
   bool near=forward-y.back()*mpp<=8.;
   if(near&&(std::abs(anchor)<.65||std::abs(anchor)>std::max(4.2,reach)))continue;
   int lo=y.front(),hi=y.back();std::vector<double>sums(hi-lo+1,0.);std::vector<int>nums(hi-lo+1,0);
   for(int k=0;k<n;++k){sums[y[k]-lo]+=x[k];nums[y[k]-lo]++;}
   std::vector<double> residual,pts;
   for(int row=hi;row>=lo;--row)if(nums[row-lo]){double col=sums[row-lo]/nums[row-lo];pts.push_back(forward-row*mpp);pts.push_back(-(lateral+col*mpp));residual.push_back(std::abs(col-(m*row+b)));}
   std::sort(residual.begin(),residual.end());double pos=.9*(residual.size()-1);int lower=int(pos);double q=residual[lower];if(lower<int(residual.size())-1)q+=(pos-lower)*(residual[lower+1]-q);
   if(q*mpp>.15)continue;
   out.push_back({std::move(x),std::move(y),std::move(pts),m,b,double(lo),double(hi),normal,double(yc)/n>.12});
  }
 }
 }
 py::list result;
 for(auto& a:out){py::array_t<double>points({py::ssize_t(a.points.size()/2),py::ssize_t(2)});std::copy(a.points.begin(),a.points.end(),points.mutable_data());py::array_t<int>x(a.x.size()),y(a.y.size());std::copy(a.x.begin(),a.x.end(),x.mutable_data());std::copy(a.y.begin(),a.y.end(),y.mutable_data());result.append(py::make_tuple(points,a.m,a.b,a.lo,a.hi,a.width,a.yellow,y,x));}
 return result;
}
py::object visible(Points points,Bytes valid,double mpp,double forward,double lateral,double span){
 auto p=points.unchecked<2>();auto v=valid.unchecked<2>();int n=p.shape(0),start=-1,best=-1,end=-1;double longest=-1,lo=0,hi=0;
 for(int i=0;i<=n;++i){bool good=false;if(i<n){double x=(-p(i,1)-lateral)/mpp,y=(forward-p(i,0))/mpp;good=x>=0&&x<v.shape(1)&&y>=0&&y<v.shape(0)&&v(int(y),int(x))>0;}
  if(good){if(start<0){start=i;lo=hi=p(i,0);}else{lo=std::min(lo,p(i,0));hi=std::max(hi,p(i,0));}}
  else if(start>=0){if(i-start>=3&&hi-lo>longest){best=start;end=i;longest=hi-lo;}start=-1;}
 }
 if(best<0||longest<span)return py::none();
 std::vector<int>order;for(int i=best;i<end;++i)order.push_back(i);std::stable_sort(order.begin(),order.end(),[&](int a,int b){return p(a,0)<p(b,0);});
 py::array_t<double>out({py::ssize_t(order.size()),py::ssize_t(2)});auto r=out.mutable_unchecked<2>();for(int i=0;i<int(order.size());++i){r(i,0)=p(order[i],0);r(i,1)=p(order[i],1);}return out;
}
double interp(const double*p,int n,double x){if(x<=p[0])return p[1];if(x>=p[2*(n-1)])return p[2*(n-1)+1];int lo=0,hi=n-1;while(hi-lo>1){int mid=(hi+lo)/2;if(p[2*mid]<=x)lo=mid;else hi=mid;}return p[2*lo+1]+(x-p[2*lo])/(p[2*hi]-p[2*lo])*(p[2*hi+1]-p[2*lo+1]);}
py::list match(py::list current,py::list previous,py::list sides,py::list old_sides,double limit){
 struct Item{double error;int i,j;};std::vector<Item>pairs;std::vector<Points> a,b;for(auto p:current)a.push_back(py::cast<Points>(p));for(auto p:previous)b.push_back(py::cast<Points>(p));
 for(int i=0;i<int(a.size());++i)for(int j=0;j<int(b.size());++j){if(py::cast<std::string>(sides[i])!=py::cast<std::string>(old_sides[j]))continue;
  auto x=a[i].data(),y=b[j].data();int nx=a[i].shape(0),ny=b[j].shape(0);double lo=std::max(x[0],y[0]),hi=std::min(x[2*(nx-1)],y[2*(ny-1)]);if(hi-lo<.20)continue;double e[8];for(int k=0;k<8;++k){double t=lo+(hi-lo)*k/7.;e[k]=std::abs(interp(x,nx,t)-interp(y,ny,t));}std::sort(e,e+8);double error=(e[3]+e[4])*.5;if(error<limit)pairs.push_back({error,i,j});
 }
 std::sort(pairs.begin(),pairs.end(),[](const Item&a,const Item&b){if(a.error!=b.error)return a.error<b.error;if(a.i!=b.i)return a.i<b.i;return a.j<b.j;});std::vector<bool>useda(a.size()),usedb(b.size());py::list result;
 for(auto pair:pairs)if(!useda[pair.i]&&!usedb[pair.j]){useda[pair.i]=usedb[pair.j]=true;result.append(py::make_tuple(pair.i,pair.j));}return result;
}

// ---- Plan 25: per-pixel parts of CarlaDetector.binary and the per-fragment edge test, with NumPy's float32
// arithmetic (a Python float scalar against a float32 array is a float32), so the masks are the same bits.
using Floats=py::array_t<float,py::array::c_style|py::array::forcecast>;
py::tuple color_planes(Bytes bev,Bytes hls,double white_min){
 auto bb=bev.request(),hb=hls.request();
 if(bb.ndim!=3||bb.shape[2]!=3||hb.shape!=bb.shape)throw std::invalid_argument("color_planes: two HxWx3 images");
 int h=bb.shape[0],w=bb.shape[1];auto p=static_cast<unsigned char*>(bb.ptr);auto q=static_cast<unsigned char*>(hb.ptr);
 Bytes white({h,w}),yellow({h,w}),level({h,w});Floats yellowness({h,w});
 auto W=white.mutable_data(),Y=yellow.mutable_data(),L=level.mutable_data();auto F=yellowness.mutable_data();
 const float k09=0.09f,k125=1.25f;
 {py::gil_scoped_release release;
 for(int i=0;i<h*w;++i){
  float b=p[3*i],g=p[3*i+1],r=p[3*i+2];int H=q[3*i],Lt=q[3*i+1],S=q[3*i+2];
  W[i]=(S<120)&&(double(Lt)>white_min);
  float gb=g-b;Y[i]=(H>=16)&&(H<=42)&&(S>20)&&(gb>18.f)&&(gb>k09*std::max(g,1.f))&&(gb>k125*(r-g));
  float yl=(r+g)/2.f-b;F[i]=yl;float lv=std::min(std::max(yl+128.f,0.f),255.f);L[i]=(unsigned char)lv;
 }}
 return py::make_tuple(white,yellow,yellowness,level);
}
Floats ridges(Floats gray,Floats smooth,std::vector<int> radii,double ridge_min,double relative_min){
 auto gb=gray.request(),sb=smooth.request();
 if(gb.ndim!=2||sb.shape!=gb.shape)throw std::invalid_argument("ridges: two HxW float32 images");
 int h=gb.shape[0],w=gb.shape[1];auto G=static_cast<float*>(gb.ptr);auto M=static_cast<float*>(sb.ptr);
 Floats response({h,w});auto R=response.mutable_data();const float rmin=float(ridge_min),rel=float(relative_min);
 {py::gil_scoped_release release;
 std::fill(R,R+h*w,0.f);
 for(int radius:radii)for(int y=0;y<h;++y){const float*m=M+y*w;const float*g=G+y*w;float*r=R+y*w;
  for(int x=0;x<w;++x){float side=std::max(m[std::max(0,x-radius)],m[std::min(w-1,x+radius)]);float ridge=g[x]-side;
   float v=(ridge>=rmin&&ridge>=rel*std::max(side,25.f))?ridge:0.f;r[x]=std::max(r[x],v);}}
 }
 return response;
}
// NumPy's pairwise float32 sum (numpy/core/src/umath/loops_utils.h.src), so the edge sums are the same.
static float pairwise(const float*a,int n){
 if(n<8){float s=0.f;for(int i=0;i<n;++i)s+=a[i];return s;}
 if(n<=128){float r[8];for(int k=0;k<8;++k)r[k]=a[k];int i;for(i=8;i<n-(n%8);i+=8)for(int k=0;k<8;++k)r[k]+=a[i+k];
  float res=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));for(;i<n;++i)res+=a[i];return res;}
 int n2=n/2;n2-=n2%8;return pairwise(a,n2)+pairwise(a+n2,n-n2);
}
std::vector<bool> edge_like(Floats gray,py::list fragments,int offset){
 auto gb=gray.request();int h=gb.shape[0],w=gb.shape[1];auto G=static_cast<float*>(gb.ptr);std::vector<bool> out;
 std::vector<float> row[5];
 for(auto item:fragments){
  auto t=py::cast<py::tuple>(item);auto ya=py::cast<py::array_t<int,py::array::c_style|py::array::forcecast>>(t[0]),xa=py::cast<py::array_t<int,py::array::c_style|py::array::forcecast>>(t[1]);
  auto Y=ya.data(),X=xa.data();int n=ya.size();for(auto&r:row)r.clear();
  for(int i=0;i<n;++i){int x=X[i];if(!(x-offset-1>=0&&x+offset+1<w))continue;const float*g=G+Y[i]*w;
   row[0].push_back(g[x-offset]);row[1].push_back(g[x-offset-1]);row[2].push_back(g[x+offset]);row[3].push_back(g[x+offset+1]);row[4].push_back(g[x]);}
  int count=row[0].size();if(count<3){out.push_back(false);continue;}
  float s[5];for(int k=0;k<5;++k)s[k]=pairwise(row[k].data(),count);
  double left=double(s[0]+s[1])/(2*count),right=double(s[2]+s[3])/(2*count),center=double(s[4])/count,asym=std::abs(left-right);
  out.push_back(asym>=20.&&center-std::max(left,right)<asym/1.5);
 }
 (void)h;return out;
}

// ---- Plan 25: per-lane loops of TemporalDetector.process with NumPy's exact arithmetic (np.linspace,
// np.interp as numpy/core/src/multiarray/compiled_base.c, the median of core.median).
static double np_interp(const double*xp,const double*fp,int n,double x,int stride){
 // xp/fp: n values with the given stride (points arrays are N x 2)
 if(x<xp[0])return fp[0];
 if(x>xp[(n-1)*stride])return fp[(n-1)*stride];
 if(x==xp[(n-1)*stride])return fp[(n-1)*stride];
 int lo=0,hi=n-1;  // largest j with xp[j] <= x (x < xp[n-1] here)
 while(hi-lo>1){int mid=(lo+hi)/2;if(xp[mid*stride]<=x)lo=mid;else hi=mid;}
 int j=lo;double xj=xp[j*stride],yj=fp[j*stride];
 if(xj==x)return yj;
 double xk=xp[(j+1)*stride],yk=fp[(j+1)*stride],slope=(yk-yj)/(xk-xj),r=slope*(x-xj)+yj;
 if(std::isnan(r)){r=slope*(x-xk)+yk;if(std::isnan(r)&&yj==yk)r=yj;}
 return r;
}
static double np_median(std::vector<double>v){
 int n=v.size();if(!n)return NAN;int k=n/2;
 for(double a:v)if(std::isnan(a))return NAN;
 if(n%2){std::nth_element(v.begin(),v.begin()+k,v.end());return v[k];}
 std::nth_element(v.begin(),v.begin()+k,v.end());double hi=v[k];double lo=*std::max_element(v.begin(),v.begin()+k);return (lo+hi)/2;
}
// The observed lane against the track it continues: the median distance at 12 stations over the overlap, and
// (when below limit) the overlap moved toward the track by 1-weight, in place.  Returns the median (inf: no overlap).
double track_match(py::array_t<double,py::array::c_style> a,Points b,double limit,double weight){
 auto A=a.mutable_unchecked<2>();auto B=b.unchecked<2>();int na=A.shape(0),nb=B.shape(0);
 if(na<1||nb<1)return INFINITY;
 double lo=std::max(A(0,0),B(0,0)),hi=std::min(A(na-1,0),B(nb-1,0));
 if(!(hi>lo))return INFINITY;
 const double*ap=a.data();const double*bp=b.data();std::vector<double>e(12);double step=(hi-lo)/11.;
 for(int i=0;i<12;++i){double x=i==11?hi:(step==0?double(i)/11.*(hi-lo):double(i)*step)+(i==11?0.:lo);
  if(i==11)x=hi;e[i]=std::abs(np_interp(ap,ap+1,na,x,2)-np_interp(bp,bp+1,nb,x,2));}
 double error=np_median(e);
 if(error<limit){double*w=a.mutable_data();
  for(int i=0;i<na;++i){double x=w[2*i];if(x>=lo&&x<=hi)w[2*i+1]=weight*w[2*i+1]+(1-weight)*np_interp(bp,bp+1,nb,x,2);}}
 return error;
}
// A lane against its side's confirmed boundary: the median distance over the boundary's span (inf: < 3 points).
double corridor_error(Points p,Points b){
 auto P=p.unchecked<2>();auto B=b.unchecked<2>();int n=P.shape(0),nb=B.shape(0);if(nb<1)return INFINITY;
 const double*bp=b.data();std::vector<double>d;
 for(int i=0;i<n;++i){double x=P(i,0);if(x>=B(0,0)&&x<=B(nb-1,0))d.push_back(std::abs(P(i,1)-np_interp(bp,bp+1,nb,x,2)));}
 if(d.size()<3)return INFINITY;return np_median(d);
}

// points (x forward, y left) carried by a BEV pixel affine: TemporalDetector.metric(pixel(points) @ M[:, :2].T + M[:, 2]).
// The 2-term dot products as OpenBLAS's dgemm computes them on x86-64 with FMA: fma(b, m1, a*m0).
py::array_t<double> reproject(Points points,Points matrix,double mpp,double forward,double lateral){
 auto P=points.unchecked<2>();auto M=matrix.unchecked<2>();int n=P.shape(0);
 if(M.shape(0)!=2||M.shape(1)!=3)throw std::invalid_argument("reproject: a 2x3 matrix");
 py::array_t<double> out({py::ssize_t(n),py::ssize_t(2)});auto O=out.mutable_unchecked<2>();
 for(int i=0;i<n;++i){double px=(-P(i,1)-lateral)/mpp,py_=(forward-P(i,0))/mpp;
  double qx=std::fma(py_,M(0,1),px*M(0,0))+M(0,2),qy=std::fma(py_,M(1,1),px*M(1,0))+M(1,2);
  O(i,0)=forward-qy*mpp;O(i,1)=-(lateral+qx*mpp);}
 return out;
}
PYBIND11_MODULE(_native,m){m.def("extract_fragments",&extract,py::arg("mask"),py::arg("yellow"),py::arg("mpp"),py::arg("forward"),py::arg("lateral"),py::arg("reach")=4.2);m.def("visible",&visible);m.def("match_fragments",&match);m.def("color_planes",&color_planes);m.def("ridges",&ridges);m.def("edge_like",&edge_like);m.def("track_match",&track_match);m.def("corridor_error",&corridor_error);m.def("reproject",&reproject);}
