#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
namespace py=pybind11;
using Bytes=py::array_t<unsigned char,py::array::c_style|py::array::forcecast>;
using Points=py::array_t<double,py::array::c_style|py::array::forcecast>;
struct Fragment {std::vector<int>x,y;std::vector<double>points;double m,b,lo,hi,width;bool yellow;};
py::list extract(Bytes mask,Bytes yellow,double mpp,double forward,double lateral){
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
   if(std::abs(anchor)<.65||std::abs(anchor)>4.2)continue;
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
PYBIND11_MODULE(_native,m){m.def("extract_fragments",&extract);m.def("visible",&visible);m.def("match_fragments",&match);}
