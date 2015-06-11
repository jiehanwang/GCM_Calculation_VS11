/*******************************************************************************
Singular value decomposition program, svdcmp, from "Numerical Recipes in C"
(Cambridge Univ. Press) by W.H. Press, S.A. Teukolsky, W.T. Vetterling,
and B.P. Flannery
*******************************************************************************/
#include "stdafx.h"
#include "SVD.h"



s_SVD::s_SVD(void)
{
}


s_SVD::~s_SVD(void)
{
}

double** s_SVD::dmatrix(int nrl, int nrh, int ncl, int nch)
	/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	int i,nrow=nrh-nrl+1,ncol=nch-ncl+1;
	double **m;
	/* allocate pointers to rows */
	m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
	m += NR_END;
	m -= nrl;
	/* allocate rows and set pointers to them */
	m[nrl]=(double *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
	m[nrl] += NR_END;
	m[nrl] -= ncl;
	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
	/* return pointer to array of pointers to rows */
	return m;
}

double* s_SVD::dvector(int nl, int nh)
	/* allocate a double vector with subscript range v[nl..nh] */
{
	double *v;
	v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
	return v-nl+NR_END;
}

void s_SVD::free_dvector(double *v, int nl, int nh)
	/* free a double vector allocated with dvector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

double s_SVD::pythag(double a, double b)
	/* compute (a2 + b2)^1/2 without destructive underflow or overflow */
{
	double absa,absb;
	absa=fabs(a);
	absb=fabs(b);
	if (absa > absb) return absa*sqrt(1.0+(absb/absa)*(absb/absa));
	else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+(absa/absb)*(absa/absb)));
}


/*******************************************************************************
Given a matrix a[1..m][1..n], this routine computes its singular value
decomposition, A = U.W.VT.  The matrix U replaces a on output.  The diagonal
matrix of singular values W is output as a vector w[1..n].  The matrix V (not
the transpose VT) is output as v[1..n][1..n].
*******************************************************************************/
void s_SVD::svdcmp(double **a, int m, int n, double w[], double **v)

{
	int flag,i,its,j,jj,k,l,nm;
	double anorm,c,f,g,h,s,scale,x,y,z,*rv1;

	rv1=dvector(1,n);
	g=scale=anorm=0.0; /* Householder reduction to bidiagonal form */
	for (i=1;i<=n;i++) {
		l=i+1;
		rv1[i]=scale*g;
		g=s=scale=0.0;
		if (i <= m) {
			for (k=i;k<=m;k++) scale += fabs(a[k-1][i-1]);
			if (scale) {
				for (k=i;k<=m;k++) {
					a[k-1][i-1] /= scale;
					s += a[k-1][i-1]*a[k-1][i-1];
				}
				f=a[i-1][i-1];
				g = -SIGN(sqrt(s),f);
				h=f*g-s;
				a[i-1][i-1]=f-g;
				for (j=l;j<=n;j++) {
					for (s=0.0,k=i;k<=m;k++) s += a[k-1][i-1]*a[k-1][j-1];
					f=s/h;
					for (k=i;k<=m;k++) a[k-1][j-1] += f*a[k-1][i-1];
				}
				for (k=i;k<=m;k++) a[k-1][i-1] *= scale;
			}
		}
		w[i-1]=scale *g;
		g=s=scale=0.0;
		if (i <= m && i != n) {
			for (k=l;k<=n;k++) scale += fabs(a[i-1][k-1]);
			if (scale) {
				for (k=l;k<=n;k++) {
					a[i-1][k-1] /= scale;
					s += a[i-1][k-1]*a[i-1][k-1];
				}
				f=a[i-1][l-1];
				g = -SIGN(sqrt(s),f);
				h=f*g-s;
				a[i-1][l-1]=f-g;
				for (k=l;k<=n;k++) rv1[k]=a[i-1][k-1]/h;
				for (j=l;j<=m;j++) {
					for (s=0.0,k=l;k<=n;k++) s += a[j-1][k-1]*a[i-1][k-1];
					for (k=l;k<=n;k++) a[j-1][k-1] += s*rv1[k];
				}
				for (k=l;k<=n;k++) a[i-1][k-1] *= scale;
			}
		}
		anorm = DMAX(anorm,(fabs(w[i-1])+fabs(rv1[i])));
	}
	for (i=n;i>=1;i--) { /* Accumulation of right-hand transformations. */
		if (i < n) {
			if (g) {
				for (j=l;j<=n;j++) /* Double division to avoid possible underflow. */
					v[j-1][i-1]=(a[i-1][j-1]/a[i-1][l-1])/g;
				for (j=l;j<=n;j++) {
					for (s=0.0,k=l;k<=n;k++) s += a[i-1][k-1]*v[k-1][j-1];
					for (k=l;k<=n;k++) v[k-1][j-1] += s*v[k-1][i-1];
				}
			}
			for (j=l;j<=n;j++) v[i-1][j-1]=v[j-1][i-1]=0.0;
		}
		v[i-1][i-1]=1.0;
		g=rv1[i];
		l=i;
	}
	for (i=IMIN(m,n);i>=1;i--) { /* Accumulation of left-hand transformations. */
		l=i+1;
		g=w[i-1];
		for (j=l;j<=n;j++) a[i-1][j-1]=0.0;
		if (g) {
			g=1.0/g;
			for (j=l;j<=n;j++) {
				for (s=0.0,k=l;k<=m;k++) s += a[k-1][i-1]*a[k-1][j-1];
				f=(s/a[i-1][i-1])*g;
				for (k=i;k<=m;k++) a[k-1][j-1] += f*a[k-1][i-1];
			}
			for (j=i;j<=m;j++) a[j-1][i-1] *= g;
		} else for (j=i;j<=m;j++) a[j-1][i-1]=0.0;
		++a[i-1][i-1];
	}
	for (k=n;k>=1;k--) { /* Diagonalization of the bidiagonal form. */
		for (its=1;its<=30;its++) {
			flag=1;
			for (l=k;l>=1;l--) { /* Test for splitting. */
				nm=l-1; /* Note that rv1[1] is always zero. */
				if ((double)(fabs(rv1[l])+anorm) == anorm) {
					flag=0;
					break;
				}
				if ((double)(fabs(w[nm-1])+anorm) == anorm) break;
			}
			if (flag) {
				c=0.0; /* Cancellation of rv1[l], if l > 1. */
				s=1.0;
				for (i=l;i<=k;i++) {
					f=s*rv1[i];
					rv1[i]=c*rv1[i];
					if ((double)(fabs(f)+anorm) == anorm) break;
					g=w[i-1];
					h=pythag(f,g);
					w[i-1]=h;
					h=1.0/h;
					c=g*h;
					s = -f*h;
					for (j=1;j<=m;j++) {
						y=a[j-1][nm-1];
						z=a[j-1][i-1];
						a[j-1][nm-1]=y*c+z*s;
						a[j-1][i-1]=z*c-y*s;
					}
				}
			}
			z=w[k-1];
			if (l == k) { /* Convergence. */
				if (z < 0.0) { /* Singular value is made nonnegative. */
					w[k-1] = -z;
					for (j=1;j<=n;j++) v[j-1][k-1] = -v[j-1][k-1];
				}
				break;
			}
			if (its == 30) printf("no convergence in 30 svdcmp iterations");
			x=w[l-1]; /* Shift from bottom 2-by-2 minor. */
			nm=k-1;
			y=w[nm-1];
			g=rv1[nm];
			h=rv1[k];
			f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
			g=pythag(f,1.0);
			f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
			c=s=1.0; /* Next QR transformation: */
			for (j=l;j<=nm;j++) {
				i=j+1;
				g=rv1[i];
				y=w[i-1];
				h=s*g;
				g=c*g;
				z=pythag(f,h);
				rv1[j]=z;
				c=f/z;
				s=h/z;
				f=x*c+g*s;
				g = g*c-x*s;
				h=y*s;
				y *= c;
				for (jj=1;jj<=n;jj++) {
					x=v[jj-1][j-1];
					z=v[jj-1][i-1];
					v[jj-1][j-1]=x*c+z*s;
					v[jj-1][i-1]=z*c-x*s;
				}
				z=pythag(f,h);
				w[j-1]=z; /* Rotation can be arbitrary if z = 0. */
				if (z) {
					z=1.0/z;
					c=f*z;
					s=h*z;
				}
				f=c*g+s*y;
				x=c*y-s*g;
				for (jj=1;jj<=m;jj++) {
					y=a[jj-1][j-1];
					z=a[jj-1][i-1];
					a[jj-1][j-1]=y*c+z*s;
					a[jj-1][i-1]=z*c-y*s;
				}
			}
			rv1[l]=0.0;
			rv1[k]=f;
			w[k-1]=x;
		}
	}
	free_dvector(rv1,1,n);
}