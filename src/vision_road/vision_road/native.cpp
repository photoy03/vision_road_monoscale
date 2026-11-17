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

// ---- Plan 26: CarlaDetector.coherent_paths' best_path DP (same order of additions and comparisons).
// ok, weight: n x n links k -> j (k < j); root: j may start a chain for this side; excluded: j is left out.
py::tuple chain_dp(py::array_t<bool,py::array::c_style|py::array::forcecast> ok,Points weight,Points base,
                   py::array_t<bool,py::array::c_style|py::array::forcecast> root,py::array_t<bool,py::array::c_style|py::array::forcecast> excluded,Points length){
 int n=base.shape(0);auto O=ok.data();auto W=weight.data();auto B=base.data();auto R=root.data();auto X=excluded.data();auto L=length.data();
 py::array_t<double> scores(n),paint(n);py::array_t<long> parent(n),origin(n);
 auto S=scores.mutable_data();auto P=paint.mutable_data();auto Pa=parent.mutable_data();auto Or=origin.mutable_data();
 const double NEG=-INFINITY;
 for(int j=0;j<n;++j){S[j]=NEG;Pa[j]=-1;Or[j]=j;P[j]=L[j];}
 for(int j=0;j<n;++j){
  if(X[j])continue;
  double best=R[j]?B[j]:NEG;int choice=-1;
  for(int k=0;k<j;++k){if(!O[k*n+j])continue;if(S[k]==NEG)continue;double value=S[k]+W[k*n+j];if(value>best){best=value;choice=k;}}
  S[j]=best;
  if(choice>=0){Pa[j]=choice;Or[j]=Or[choice];P[j]=P[choice]+L[j];}
 }
 return py::make_tuple(scores,parent,origin,paint);
}

// ---- Plan 26: CarlaDetector.pair_vote's pair test (same order; median and np.interp as above).
// Returns (i, j, d, hi-lo) for every accepted pair, in the Python loop's order.
py::list pair_tests(py::list points,std::vector<double> first,std::vector<double> last,std::vector<double> slope,std::vector<double> intercept,
                    std::vector<bool> wide,double gap){
 int n=points.size();std::vector<Points> P;for(auto p:points)P.push_back(py::cast<Points>(p));py::list out;
 for(int i=0;i<n;++i){if(wide[i])continue;const double*a=P[i].data();int na=P[i].shape(0);
  for(int j=i+1;j<n;++j){if(wide[j])continue;
   double lo=std::max(first[i],first[j]),hi=std::min(last[i],last[j]);if(hi-lo<1.)continue;
   double middle=(lo+hi)/2;if(std::abs(std::abs((slope[i]-slope[j])*middle+intercept[i]-intercept[j])-gap)>.15)continue;
   const double*b=P[j].data();int nb=P[j].shape(0);std::vector<double> v;
   for(int k=0;k<na;++k){double x=a[2*k];if(x>=lo&&x<=hi)v.push_back(a[2*k+1]-np_interp(b,b+1,nb,x,2));}
   double d=np_median(v);
   if(std::abs(std::abs(d)-gap)>.08)continue;  // NaN passes, as in Python
   out.append(py::make_tuple(i,j,d,hi-lo));
  }}
 return out;
}

// ---- Plan 26: the paint / visible profile sampled along a line on the BEV (marking_style.classify_lane and
// TemporalDetector._accumulate_style), with np.arange's values (start + i*((start+step)-start), length
// ceil((stop-start)/step)), np.interp and np.rint (nearbyint, ties to even).
static std::vector<double> np_arange(double start,double stop,double step){
 // numpy PyArray_Arange: element 1 is start+step, the rest start+i*delta with delta=(start+step)-start
 std::vector<double> v;double span=stop-start,t=span/step;long n;
 if(t==0.&&span!=0.)n=std::signbit(t)?0:1;else{double len=std::ceil(t);if(!(len>0))return v;n=long(len);}
 if(n<=0)return v;v.resize(n);v[0]=start;if(n==1)return v;v[1]=start+step;double delta=v[1]-start;for(long i=2;i<n;++i)v[i]=start+double(i)*delta;return v;
}
static void sample_masks(const std::vector<double>&fx,const std::vector<double>&fy,Bytes binary,Bytes valid,double mpp,double forward,double lateral,
                         py::array_t<bool>&paint,py::array_t<bool>&seen){
 auto bb=binary.request();int h=bb.shape[0],w=bb.shape[1];auto Bp=static_cast<unsigned char*>(bb.ptr);auto Vp=valid.data();
 int n=fx.size(),width=std::max(1,int(std::nearbyint(.12/mpp)));paint=py::array_t<bool>(n);seen=py::array_t<bool>(n);auto P=paint.mutable_data();auto S=seen.mutable_data();
 for(int i=0;i<n;++i){long col=long(std::nearbyint((-fy[i]-lateral)/mpp)),row=long(std::nearbyint((forward-fx[i])/mpp));
  bool inside=row>=0&&row<h&&col>=width&&col<w-width;long rr=std::min(std::max(row,0L),long(h-1));bool all=true,any=false;
  for(int d=-width;d<=width;++d){long cc=std::min(std::max(col+d,0L),long(w-1));all=all&&Vp[rr*w+cc]>0;any=any||Bp[rr*w+cc]>0;}
  S[i]=inside&&all;P[i]=inside&&any;}
}
// classify_lane: (paint, visible) along the lane's arc every 0.1 m, or None when the arc is shorter than one step.
py::object lane_profile(Points points,Bytes binary,Bytes valid,double mpp,double forward,double lateral){
 auto p=points.unchecked<2>();int n=p.shape(0);std::vector<double> arc(n,0.);
 for(int i=1;i<n;++i){double dx=p(i,0)-p(i-1,0),dy=p(i,1)-p(i-1,1);arc[i]=arc[i-1]+std::sqrt(dx*dx+dy*dy);}
 auto sample=np_arange(0.,arc[n-1],.10);if(sample.empty())return py::none();
 std::vector<double> xs(n),ys(n);for(int i=0;i<n;++i){xs[i]=p(i,0);ys[i]=p(i,1);}
 std::vector<double> fx(sample.size()),fy(sample.size());
 for(size_t i=0;i<sample.size();++i){fx[i]=np_interp(arc.data(),xs.data(),n,sample[i],1);fy[i]=np_interp(arc.data(),ys.data(),n,sample[i],1);}
 py::array_t<bool> paint,seen;sample_masks(fx,fy,binary,valid,mpp,forward,lateral,paint,seen);return py::make_tuple(paint,seen);
}
// _accumulate_style: x = arange(start, stop, 0.1) along forward, y = interp on the sorted line; (x, seen, paint).
py::tuple forward_profile(Points sorted,double start,double stop,Bytes binary,Bytes valid,double mpp,double forward,double lateral){
 auto x=np_arange(start,stop,.1);const double*q=sorted.data();int n=sorted.shape(0);std::vector<double> y(x.size());
 for(size_t i=0;i<x.size();++i)y[i]=np_interp(q,q+1,n,x[i],2);
 py::array_t<bool> paint,seen;sample_masks(x,y,binary,valid,mpp,forward,lateral,paint,seen);
 py::array_t<double> xa(x.size());std::copy(x.begin(),x.end(),xa.mutable_data());return py::make_tuple(xa,seen,paint);
}

// ---- Plan 26: the iteratively reweighted least squares of CarlaDetector.spine and pair_vote.  Householder QR
// instead of LAPACK's SVD (np.linalg.lstsq): the same solution up to rounding (not bit for bit).  Returns false
// when the system is (nearly) rank deficient, where lstsq's minimum-norm answer differs: Python then does it.
static bool least_squares(std::vector<double> A,std::vector<double> b,int m,int n,std::vector<double>&x){
 // A: m x n row-major
 std::vector<double> diag(n);double amax=0;for(double v:A)amax=std::max(amax,std::abs(v));
 for(int k=0;k<n;++k){
  double norm=0;for(int i=k;i<m;++i)norm+=A[i*n+k]*A[i*n+k];norm=std::sqrt(norm);
  if(norm<=1e-13*std::max(amax,1e-300)*std::max(m,n))return false;
  double alpha=A[k*n+k]>0?-norm:norm;double v0=A[k*n+k]-alpha;A[k*n+k]=v0;
  double vnorm2=v0*v0;for(int i=k+1;i<m;++i)vnorm2+=A[i*n+k]*A[i*n+k];
  if(vnorm2>0){
   for(int j=k+1;j<n;++j){double dot=0;for(int i=k;i<m;++i)dot+=A[i*n+k]*A[i*n+j];double f=2*dot/vnorm2;for(int i=k;i<m;++i)A[i*n+j]-=f*A[i*n+k];}
   double dot=0;for(int i=k;i<m;++i)dot+=A[i*n+k]*b[i];double f=2*dot/vnorm2;for(int i=k;i<m;++i)b[i]-=f*A[i*n+k];
  }
  diag[k]=alpha;
 }
 x.assign(n,0.);for(int k=n-1;k>=0;--k){double sum=b[k];for(int j=k+1;j<n;++j)sum-=A[k*n+j]*x[j];x[k]=sum/diag[k];}
 return true;
}
// spine's robust(): t = x/10, A = vander(t, degree+1); 5 reweightings.  (coef, residual, weights) or None.
py::object spine_fit(Points t,Points center,Points weights0,Points near,int degree,double ridge){
 int m=t.shape(0),n=degree+1;const double*T=t.data();const double*C=center.data();const double*N=near.data();
 std::vector<double> w(weights0.data(),weights0.data()+m),coef,res(m);std::vector<double> V(m*n);
 for(int i=0;i<m;++i){double p=1;for(int k=n-1;k>=0;--k){V[i*n+k]=p;p*=T[i];}}
 int rows=m+(ridge>0);
 for(int it=0;it<5;++it){
  std::vector<double> A(rows*n,0.),b(rows,0.);
  for(int i=0;i<m;++i){double wi=w[i]*N[i];for(int k=0;k<n;++k)A[i*n+k]=V[i*n+k]*wi;b[i]=C[i]*wi;}
  if(ridge>0)A[m*n]=std::sqrt(ridge);
  if(!least_squares(A,b,rows,n,coef))return py::none();
  for(int i=0;i<m;++i){double f=0;for(int k=0;k<n;++k)f+=V[i*n+k]*coef[k];res[i]=std::abs(C[i]-f);w[i]=std::min(1.,.20/std::max(res[i],.001));}
 }
 py::array_t<double> c(n),r(m),ww(m);std::copy(coef.begin(),coef.end(),c.mutable_data());std::copy(res.begin(),res.end(),r.mutable_data());std::copy(w.begin(),w.end(),ww.mutable_data());
 return py::make_tuple(c,r,ww);
}
// pair_vote's line through the labelled centres: 3 reweighted polyfits of degree 0 or 1 (np.polyfit's column scaling).
py::object pair_fit(Points xs,Points cs,int degree){
 int m=xs.shape(0),n=degree+1;const double*X=xs.data();const double*C=cs.data();std::vector<double> w(m,1.),fit(n);
 for(int it=0;it<3;++it){
  std::vector<double> A(m*n),b(m),scale(n,0.);
  for(int i=0;i<m;++i){double p=1;for(int k=n-1;k>=0;--k){A[i*n+k]=p*w[i];p*=X[i];}b[i]=C[i]*w[i];}
  for(int k=0;k<n;++k){for(int i=0;i<m;++i)scale[k]+=A[i*n+k]*A[i*n+k];scale[k]=std::sqrt(scale[k]);if(scale[k]==0)return py::none();for(int i=0;i<m;++i)A[i*n+k]/=scale[k];}
  std::vector<double> c;if(!least_squares(A,b,m,n,c))return py::none();
  for(int k=0;k<n;++k)fit[k]=c[k]/scale[k];
  for(int i=0;i<m;++i){double f=0;for(int k=0;k<n;++k)f=f*X[i]+fit[k];w[i]=std::min(1.,.05/std::max(std::abs(C[i]-f),.001));}
 }
 py::array_t<double> out(n);std::copy(fit.begin(),fit.end(),out.mutable_data());return out;
}

// ---- Plan 26 (bit for bit): CarlaDetector.width_error and recall_pairs.
static std::vector<double> np_linspace(double lo,double hi,int num){
 std::vector<double> v(num);int div=num-1;double delta=hi-lo,step=delta/div;
 for(int i=0;i<num;++i)v[i]=(step==0?(double(i)/div)*delta:double(i)*step)+lo;v[num-1]=hi;return v;
}
py::object width_gap(Points points,Points ref,bool right,double lane_width){
 int n=points.shape(0),m=ref.shape(0);const double*p=points.data();const double*r=ref.data();
 double lo=std::max(p[0],r[0]),hi=std::min(std::min(p[2*(n-1)],r[2*(m-1)]),lo+4.);
 if(hi<lo)return py::none();
 auto x=np_linspace(lo,hi,5);std::vector<double> d(5);
 for(int i=0;i<5;++i){double own=np_interp(p,p+1,n,x[i],2),them=np_interp(r,r+1,m,x[i],2);d[i]=right?them-own:own-them;}
 return py::float_(std::abs(np_median(d)-lane_width));
}
// recall_pairs' remembered centres per 0.2 m cell and the labels they give: [(i, label)] in node order.
py::list recall_labels(Points world,double px,double py_,double co,double si,int cells_n,py::list points,std::vector<bool> skip,double half){
 int m=world.shape(0);const double*W=world.data();std::vector<long> count(cells_n,0);std::vector<double> total(cells_n,0.);
 for(int i=0;i<m;++i){double dx=W[2*i]-px,dy=W[2*i+1]-py_,ahead=dx*co+dy*si,lat=(-dx)*si+dy*co;double f=std::floor(ahead/.2);
  if(!(f>=0&&f<cells_n))continue;long c=long(f);count[c]++;total[c]+=lat;}
 py::list out;int k=0;
 for(auto item:points){int i=k++;if(skip[i])continue;auto P=py::cast<Points>(item);const double*q=P.data();int n=P.shape(0);std::vector<double> d;
  for(int j=0;j<n;++j){double f=std::floor(q[2*j]/.2);long c=f<0?0:(f>cells_n-1?cells_n-1:long(f));if(count[c]>0)d.push_back(total[c]/double(count[c])-q[2*j+1]);}
  if(d.size()<3)continue;double v=np_median(d);
  if(std::abs(std::abs(v)-half)<=half*.6)out.append(py::make_tuple(i,v>0?-1.:1.));}
 return out;
}
PYBIND11_MODULE(_native,m){m.def("extract_fragments",&extract,py::arg("mask"),py::arg("yellow"),py::arg("mpp"),py::arg("forward"),py::arg("lateral"),py::arg("reach")=4.2);m.def("visible",&visible);m.def("match_fragments",&match);m.def("color_planes",&color_planes);m.def("ridges",&ridges);m.def("edge_like",&edge_like);m.def("track_match",&track_match);m.def("corridor_error",&corridor_error);m.def("reproject",&reproject);m.def("chain_dp",&chain_dp);m.def("pair_tests",&pair_tests);m.def("lane_profile",&lane_profile);m.def("forward_profile",&forward_profile);m.def("spine_fit",&spine_fit);m.def("pair_fit",&pair_fit);m.def("width_gap",&width_gap);m.def("recall_labels",&recall_labels);}
