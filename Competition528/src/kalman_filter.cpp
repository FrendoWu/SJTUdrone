#include "kalman_filter.h"


//Union prototype for conversion from int to float
/*union就是里面的成员共用同一个存储空间，这个存储空间
的大小与union中占用空间最大的一个成员相同。*/
typedef union  
{
	int ii;
	float ff;

}f2i;


float kalman(float x_in, prev_states prev[], int i, float Q, float R)
{

	float x, y, y_prev, p, p_prev, K, Qf, Rf;
	f2i temp;


	//Measured value
	//temp.ii = x_in;
	//x = temp.ff;
	x= x_in;

	//Previous values
	//temp.ii = prev[0].filtered;
	//y_prev = temp.ff;
	y_prev = prev[0].filtered;

	//temp.ii = prev[0].est_error;
	//p_prev = temp.ff;
	p_prev = prev[0].est_error;

	//Adjustin sensor variance
	Qf = Q / 10000;
	Rf = R / 100;

	//Initialize
	if (i == 0)
	{
		y = x;
		p = Rf*Rf;
	}
	else
	{
		//Filter
		p = p_prev + Qf;
		K = p / (p + Rf);
		y = y_prev + K*(x - y_prev);
		p = p*(1 - K);
	}



	//Update previous values

	//temp.ff = y;
	//prev[0].filtered = temp.ii;
	prev[0].filtered = y;

	//temp.ff = p;
	//prev[0].est_error = temp.ii;
	prev[0].est_error = p;

	return prev[0].filtered;

}
