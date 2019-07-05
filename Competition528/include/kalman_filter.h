

#ifndef FILTER_H_
#define FILTER_H_



//#define PI 3.141592


//Variable

//
//#define Qh 8		//Barometric process noise variance*10000
//#define Rh 20		//Barometric sensor noise variance*100
//
//#define Qv 9		//Pitot process noise variance*10000
//#define Rv 3		//Pitot sensor noise variance*100




typedef struct
{
	float filtered;
	float est_error;
	void start() { filtered = 0; est_error = 0; }
}prev_states;


float kalman(float x_in, prev_states prev[], int i, float Q, float R);

#endif /* FILTER_H_ */
