#include "angular_faces.h"

#include <math.h>
#include <string.h>

typedef struct { long double x[3]; } vec3;

static long double dot(vec3 a, vec3 b) {
  return a.x[0]*b.x[0]+a.x[1]*b.x[1]+a.x[2]*b.x[2];
}
static vec3 cross(vec3 a, vec3 b) {
  vec3 c={{a.x[1]*b.x[2]-a.x[2]*b.x[1],
           a.x[2]*b.x[0]-a.x[0]*b.x[2],
           a.x[0]*b.x[1]-a.x[1]*b.x[0]}};
  return c;
}
static vec3 unit(vec3 a) {
  const long double n=sqrtl(dot(a,a));
  for(int k=0;k<3;k++)a.x[k]/=n;
  return a;
}
static vec3 midpoint(vec3 a,vec3 b) {
  for(int k=0;k<3;k++)a.x[k]+=b.x[k];
  return unit(a);
}
static long double triangle_angle(vec3 a,vec3 b,vec3 c) {
  /* For unit vectors in one open hemisphere. Retaining the determinant's
   * sign makes degenerate triangles cancel rather than acquire area. */
  return 2*atan2l(dot(a,cross(b,c)),1+dot(a,b)+dot(b,c)+dot(c,a));
}
static void fill_triangle(unsigned depth,vec3 a,vec3 b,vec3 c,
                          ag_bin *bins,size_t *index) {
  if(depth) {
    const vec3 ab=midpoint(a,b),bc=midpoint(b,c),ca=midpoint(c,a);
    fill_triangle(depth-1,a,ab,ca,bins,index);
    fill_triangle(depth-1,ab,b,bc,bins,index);
    fill_triangle(depth-1,ca,bc,c,bins,index);
    fill_triangle(depth-1,ab,bc,ca,bins,index);
    return;
  }
  ag_bin *bin=&bins[(*index)++];
  const vec3 vertices[3]={a,b,c};
  for(int i=0;i<3;i++) {
    for(int k=0;k<3;k++)bin->vertices[3*i+k]=(double)vertices[i].x[k];
  }
  /* Derive shared-edge planes from the stored double vertices, so adjacent
   * bins use exactly opposite double normals on their common edge. */
  vec3 stored[3];
  for(int i=0;i<3;i++)for(int k=0;k<3;k++)stored[i].x[k]=bin->vertices[3*i+k];
  for(int i=0;i<3;i++) {
    vec3 normal=unit(cross(stored[i],stored[(i+1)%3]));
    if(dot(normal,stored[(i+2)%3])<0)
      for(int k=0;k<3;k++)normal.x[k]=-normal.x[k];
    for(int k=0;k<3;k++)bin->inward_normals[3*i+k]=(double)normal.x[k];
  }
  bin->solid_angle=(double)fabsl(triangle_angle(unit(stored[0]),unit(stored[1]),unit(stored[2])));
}

size_t ag_bin_count(unsigned level) {
  if(level>8)return 0;
  return (size_t)8<< (2*level);
}

int ag_make_bins(unsigned level,ag_bin *bins,size_t capacity) {
  const size_t count=ag_bin_count(level);
  if(!count || !bins || capacity<count)return -1;
  size_t index=0;
  for(int sx=-1;sx<=1;sx+=2)
  for(int sy=-1;sy<=1;sy+=2)
  for(int sz=-1;sz<=1;sz+=2) {
    const vec3 a={{sx,0,0}},b={{0,sy,0}},c={{0,0,sz}};
    if(sx*sy*sz>0)fill_triangle(level,a,b,c,bins,&index);
    else fill_triangle(level,a,c,b,bins,&index);
  }
  return index==count ? 0:-1;
}

static int valid_face(const double center[3],int axis,double h) {
  if(!center || axis<0 || axis>2 || !isfinite(h) || h<=0)return 0;
  for(int k=0;k<3;k++)if(!isfinite(center[k]))return 0;
  if(center[axis]==0) {
    const int u=(axis+1)%3,v=(axis+2)%3;
    if(fabs(center[u])<=h && fabs(center[v])<=h)return 0;
  }
  return 1;
}
static long double rect_corner(long double a,long double b,long double c) {
  return atan2l(b*c,a*sqrtl(a*a+b*b+c*c));
}
double ag_face_solid_angle(const double center[3],int axis,double halfwidth) {
  if(!valid_face(center,axis,halfwidth))return NAN;
  if(center[axis]==0)return 0;
  const int u=(axis+1)%3,v=(axis+2)%3;
  const long double a=fabsl(center[axis]),b=center[u],c=center[v],h=halfwidth;
  const long double area=rect_corner(a,b+h,c+h)-rect_corner(a,b-h,c+h)
    -rect_corner(a,b+h,c-h)+rect_corner(a,b-h,c-h);
  return (double)fabsl(area);
}

double ag_face_bin_solid_angle(const double center[3],int axis,
                              double halfwidth,const ag_bin *bin) {
  if(!bin || !valid_face(center,axis,halfwidth))return NAN;
  if(center[axis]==0)return 0;
  const int u=(axis+1)%3,v=(axis+2)%3;
  const int signs[4][2]={{-1,-1},{1,-1},{1,1},{-1,1}};
  vec3 poly[16],next[16],normal[3];
  int n=4,all_inside=1;
  for(int i=0;i<4;i++) {
    for(int k=0;k<3;k++)poly[i].x[k]=center[k];
    poly[i].x[u]+=signs[i][0]*(long double)halfwidth;
    poly[i].x[v]+=signs[i][1]*(long double)halfwidth;
  }
  for(int e=0;e<3;e++) {
    for(int k=0;k<3;k++) {
      if(!isfinite(bin->inward_normals[3*e+k]))return NAN;
      normal[e].x[k]=bin->inward_normals[3*e+k];
    }
    int any_inside=0;
    for(int i=0;i<4;i++) {
      const long double d=dot(normal[e],poly[i]);
      if(d>=0)any_inside=1;else all_inside=0;
    }
    if(!any_inside)return 0;
  }
  if(all_inside)return ag_face_solid_angle(center,axis,halfwidth);
  for(int e=0;e<3;e++) {
    int count=0;
    for(int i=0;i<n;i++) {
      const vec3 p=poly[i],q=poly[(i+1)%n];
      const long double dp=dot(normal[e],p),dq=dot(normal[e],q);
      const int pin=dp>=0;
      if(pin)next[count++]=p;
      /* A boundary endpoint is already emitted by its own iteration. Only
       * strict sign changes need a new point; this avoids duplicate corner
       * vertices and bounds the convex clipped polygon by seven vertices. */
      if((dp>0 && dq<0)||(dp<0 && dq>0)) {
        const long double t=dp/(dp-dq);
        vec3 at;
        for(int k=0;k<3;k++)at.x[k]=p.x[k]+t*(q.x[k]-p.x[k]);
        next[count++]=at;
      }
    }
    if(count<3)return 0;
    memcpy(poly,next,(size_t)count*sizeof(vec3));n=count;
  }
  for(int i=0;i<n;i++)poly[i]=unit(poly[i]);
  long double area=0;
  for(int i=1;i+1<n;i++)area+=triangle_angle(poly[0],poly[i],poly[i+1]);
  return (double)fabsl(area);
}
