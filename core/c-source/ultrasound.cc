/* 
 
 The ultrasound custom operations.
 
 PolarGridOp - Creates a polar grid for beamforming at the estimation point (for VFI).
 BeamformOp - Beamforms using delay, interpolate, apodize and sum in a single step.
 TimeOfFlightOp - Estimates the time of flight (delays).
 LinearInterpolationOp - Do linear interpolation of the samples. 
 NoneAndSumApodOp - Just sum the samples (boxcar apodization)
 DynamicAndSumApodOp - Dynamic apodization and sum
 EchoCancelOp - Simple samples minus mean echo cancelling
 EchoCancelThresholdOp - Echo cancelling with fourier thresholding (DOI: 10.1109/ULTSYM.2017.8091616)
 CartToPolarOp - Cartesian to polar interpolation on grid (for VFI)
 ParabolicXCorrOp - Xcorr with parabolic interpolation for velocity estimation. (Directional Beamforming)
 MCD3DOp - Estimates angle according to DOI: 10.1109/TUFFC.2016.2551689 but expanded to 3D (for VFI)
 BeamformingOp - Second implementation beamforming
 BeamformingAVXOp - Beamforming operation using AVX instructions set (for float and double)
 BeamformingGPUOp - Beamforming on the GPU using ultrasound.cu.cc kernel
 
 ----------------------------------------------------------------------------
 The MIT License (MIT)
 
 Copyright (c) 2017 Alexandra Instituttet, Aarhus, Danmark
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 ----------------------------------------------------------------------------*/

//Ultrasound ops.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <complex>
#include <fftw3.h>
#include <omp.h>


#include <iostream>
#include <valarray>
#include <ctime>


#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PI	M_PI	/* pi to machine precision, defined in math.h */
#define TWOPI	(2.0*PI)
#define TWOPIDEG (TWOPI/180.0)
#define PIDEG PI/180.0
 
#include <time.h>
#include <sys/time.h>
#include <immintrin.h>

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}



using namespace tensorflow;



REGISTER_OP("PolarGrid")
    .Attr("T: {double, float}")
    .Input("lambda: T")
    .Input("line_length: T")
    .Input("d_line: T")
    .Input("angles_theta: T")
    .Input("angles_phi: T")
    .Input("estimation_point: T")
    .Output("beamform_points: T")
    .Doc(R"doc(
Creates a polar grid for beamforming at the estimation point
)doc");

template <typename T>
class PolarGridOp : public OpKernel {
 public:
  explicit PolarGridOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {


    //Read inputs
    const Tensor& lambda_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(lambda_tensor.shape()),
                errors::InvalidArgument("focus expects a scalar."));
    auto lambda = (lambda_tensor.flat<T>())(0);

    const Tensor& line_length_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(line_length_tensor.shape()),
                errors::InvalidArgument("line_length expects a scalar."));
    auto line_length = (line_length_tensor.flat<T>())(0);

    const Tensor& d_line_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(d_line_tensor.shape()),
                errors::InvalidArgument("d_line expects a scalar."));
    auto d_line = (d_line_tensor.flat<T>())(0);

    const Tensor& angles_theta_tensor = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(angles_theta_tensor.shape()),
                errors::InvalidArgument("angles_theta expects a 1-D Vector."));
    auto angles_theta = angles_theta_tensor.flat<T>();

    const Tensor& angles_phi_tensor = context->input(4);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(angles_phi_tensor.shape()),
                errors::InvalidArgument("angles_phi expects a 1-D Vector."));
    auto angles_phi = angles_phi_tensor.flat<T>();

    //    const Tensor& estimation_point_tensor = context->input(5);
    //    OP_REQUIRES(context, TensorShapeUtils::IsVector(estimation_point_tensor.shape()),
    //                errors::InvalidArgument("point_pos expects a 1-D Vector."));
    //    auto estimation_point = estimation_point_tensor.flat<T>();

    const Tensor& estimation_point_tensor = context->input(5);
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(estimation_point_tensor.shape()),
                errors::InvalidArgument("point_pos expects a 1-D Vector or higher."));
    auto estimation_point = estimation_point_tensor.flat<T>();


    double wall_time = get_wall_time();


    int r_points_length = (int)(line_length/d_line);
    double *r_points = new double[r_points_length];
    for(int i = 0; i < r_points_length; i++)
    {
      r_points[i] = (-line_length/2.0 + i*d_line)*lambda;
    }



    // Output a float32 tensor.
    Tensor* beamform_points_tensor = NULL;

    int no_of_estimation_points = 1;
    if(estimation_point_tensor.dims() > 1)
    {
      no_of_estimation_points = estimation_point_tensor.dim_size(0);
    }
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({r_points_length*angles_theta_tensor.dim_size(0)*
			   angles_phi_tensor.dim_size(0)*no_of_estimation_points, estimation_point_tensor.dim_size(estimation_point_tensor.dims()-1)}), &beamform_points_tensor));
    auto beamform_points = beamform_points_tensor->template flat<T>();


    int phi_size = angles_phi_tensor.dim_size(0);
    int theta_size = angles_theta_tensor.dim_size(0);

    //printf("%d, %d, %d, %d\n", r_points_length, phi_size, theta_size, (int)estimation_point_tensor.dim_size(0));
    #pragma omp parallel for
    for(int p = 0; p < no_of_estimation_points; p++)
    {
      for(int i = 0; i < phi_size; i++)
      {
        double phi = angles_phi(i)*PIDEG;
	double cosphi = cos(phi);
	double sinphi = sin(phi);
	for(int j = 0; j < theta_size; j++)
        {
          double theta = (angles_theta(j)+90.0)*PIDEG;
	  double costheta = cos(theta);
	  double sintheta = sin(theta);
	  for(int k = 0; k < r_points_length; k++)
	  {
	    int index = (k + j*r_points_length + i*theta_size*r_points_length + p*phi_size*theta_size*r_points_length)*3;
	    double r_point = r_points[k];
	    double rcost = r_point * costheta;
	    beamform_points(index) = rcost * cosphi + estimation_point(p*3);
	    beamform_points(index+1) = rcost * sinphi + estimation_point(p*3+1);
	    beamform_points(index+2) = r_point * sintheta + estimation_point(p*3+2);
	  //if(i == phi_size - 2) printf("%e, %e, %e, %e, %e, %e\n", phi, theta, r_point, rcost*cos(phi), rcost*sin(phi), r_point*sin(theta));
	  }
	}
      }
    }
    //for(int i = 0; i < 100; i++)
    //  {
    //	printf("Bf: %e, %e, %e\n", beamform_points(i*3), beamform_points(i*3+1), beamform_points(i*3+2));
    //  }

    std::cout << "Polar_grid: " << (get_wall_time() - wall_time)*1000.0 << " ms" << std::endl;

  }
};

#define REGISTER_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(						 \
      Name("PolarGrid").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      PolarGridOp<type>)

REGISTER_KERNEL(double);
REGISTER_KERNEL(float);

#undef REGISTER_KERNEL



REGISTER_OP("Beamform")
    .Attr("T: {double, float}")
    .Input("focus: T")
    .Input("center_focus: T")
    .Input("speed_of_sound: T")
    .Input("t_start: T")
    .Input("t_start_data: T")
    .Input("f_sampling: T")
    .Input("f_number: T")
    .Input("element_pos: T")
    .Input("point_pos: T")
    .Input("samples: T")
    .Output("bf_samples: T")
    .Doc(R"doc(
Beamforming
)doc");

template <typename T>
class BeamformOp : public OpKernel {
 public:
  explicit BeamformOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {


    //Read inputs
    const Tensor& focus_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(focus_tensor.shape()),
                errors::InvalidArgument("focus expects a 1-D vector."));
    auto focus = focus_tensor.flat<T>();

    const Tensor& center_focus_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(center_focus_tensor.shape()),
                errors::InvalidArgument("center_focus expects a 1-D vector."));
    auto center_focus = center_focus_tensor.flat<T>();

    const Tensor& speed_of_sound_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(speed_of_sound_tensor.shape()),
                errors::InvalidArgument("speed_of_sound expects a scalar."));
    auto speed_of_sound = speed_of_sound_tensor.flat<T>();

    const Tensor& t_start_tensor = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(t_start_tensor.shape()),
                errors::InvalidArgument("t_start expects a scalar."));
    auto t_start = t_start_tensor.flat<T>();

    const Tensor& t_start_data_tensor = context->input(4);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(t_start_data_tensor.shape()),
                errors::InvalidArgument("t_start_data expects a scalar."));
    auto t_start_data = t_start_data_tensor.flat<T>();


    const Tensor& f_sampling_tensor = context->input(5);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(f_sampling_tensor.shape()),
                errors::InvalidArgument("f_sampling expects a scalar."));
    auto f_sampling = f_sampling_tensor.flat<T>();

    const Tensor& f_number_tensor = context->input(6);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(f_number_tensor.shape()),
                errors::InvalidArgument("f_number expects a scalar."));
    auto f_number = f_number_tensor.flat<T>();


    const Tensor& element_pos_tensor = context->input(7);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(element_pos_tensor.shape()),
                errors::InvalidArgument("element_pos expects a matrix."));
    auto element_pos = element_pos_tensor.flat<T>();

    const Tensor& point_pos_tensor = context->input(8);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(point_pos_tensor.shape()),
                errors::InvalidArgument("point_pos expects a matrix."));
    auto point_pos = point_pos_tensor.flat<T>();

    const Tensor& samples_tensor = context->input(9);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(samples_tensor.shape()),
                errors::InvalidArgument("samples expects a matrix."));
    auto samples = samples_tensor.flat<T>();



    double wall_time = get_wall_time();

    // Distance center focus to center
    T tof_cf = 0;
    for (int i = 0; i < focus.size(); i++)
    {
      tof_cf += (focus(i) - center_focus(i)) * (focus(i) - center_focus(i));
    }
    tof_cf = sqrtf(tof_cf);


    //printf("point_pos: %d, %d\n", (int)point_pos_tensor.dim_size(0), (int)point_pos_tensor.dim_size(1));

	  

    // Output a float32 tensor.
    Tensor* bf_samples_tensor = NULL;
    int output_dimensions = 1;
    if(samples_tensor.dims() > 2)
    {
      output_dimensions = samples_tensor.dim_size(0);
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({samples_tensor.dim_size(0), point_pos_tensor.dim_size(0)}), &bf_samples_tensor));
    }
    else
    {
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({point_pos_tensor.dim_size(0)}), &bf_samples_tensor));
    }
    auto bf_samples = bf_samples_tensor->template flat<T>();



    //Loop through all points and get distance to virtual source
    const int points = point_pos_tensor.dim_size(0);
    const int elements = element_pos_tensor.dim_size(0);
    const int Dims = point_pos_tensor.dim_size(1);
    auto inv_sof = 1.0/speed_of_sound(0);

    const int sample_points = samples_tensor.dim_size(samples_tensor.dims()-2);
    const int sample_elements = samples_tensor.dim_size(samples_tensor.dims()-1);


    printf("points: %d, elements: %d\n", points, elements);

    auto max_index = (int)samples_tensor.dim_size(samples_tensor.dims()-2) - 2;


    #pragma omp parallel for
    for (int i = 0; i < points; i++) 
    {


      //Calc dist from point to VS(focus)
      T tof_tx = 0;
      for (int k = 0; k < Dims; k++)
      {
        tof_tx += (focus(k) - point_pos(i*Dims+k)) * (focus(k) - point_pos(i*Dims+k));
      }
      tof_tx = sqrtf(tof_tx);


      //FIND APOD VALUES
      T *ref_xyz = new T[Dims];
      T ref_dist = 1000000;
      T *tof_tmp = new T[elements];
      for(int j = 0; j < elements; j++)
      {
        //Calc dist from point to element
        T tof_rx = 0;
        for (int k = 0; k < Dims; k++)
        {
          tof_rx += (element_pos(j*Dims+k) - point_pos(i*Dims+k)) * (element_pos(j*Dims+k) - point_pos(i*Dims+k));
        }
        tof_rx = sqrtf(tof_rx);
	tof_tmp[j] = (tof_rx+tof_tx+tof_cf)*inv_sof - t_start(0);

        //Save closest distance to element and vector from element to bf_point
        if(tof_rx < ref_dist)
        { 
          ref_dist = tof_rx;
	  for (int k = 0; k < Dims; k++)
	  {
            ref_xyz[k] = element_pos(j*Dims+k) - point_pos(i*Dims+k);
	  }
        }

      }
      T* norm_ref = new T[Dims];
      for (int k = 0; k < Dims; k++)
      {
        norm_ref[k] = point_pos(i*Dims+k) + ref_xyz[k] / ref_dist;
      }

      //Calc distance to line for all elements

      ref_dist = ref_dist/f_number(0) * 0.5;
      //      if( i == 0)
      //	printf("i: %d, ref_dist: %f\n", i, ref_dist);


      //END FIND APOD VALUES





      for (int j = 0; j < elements; j++) 
      {
	T tof_idx_weight = (tof_tmp[j]-t_start_data(0))*f_sampling(0);
	int tof_idx = round(tof_idx_weight);
	tof_idx_weight = tof_idx_weight - tof_idx;
	tof_idx = std::max(0, std::min(tof_idx, max_index));
	int tof_idx_plus = tof_idx + 1;

        //if(i == 0 && j == 0)
	//  printf("tof_tmp: %e, %d, %d, %e, %e\n", tof_tmp, tof_idx, tof_idx_plus, t_start_data(0), f_sampling(0));

	//BEGIN APOD WEIGHT
        T* v1 = new T[Dims];
        T* v2 = new T[Dims];
        for (int k = 0; k < Dims; k++)
        {
          v1[k] = element_pos(j*Dims+k) - norm_ref[k];
          v2[k] = element_pos(j*Dims+k) - point_pos(i*Dims+k);
        }
        T* cross = new T[Dims];
	for( int k = 0; k < Dims; k++)
        {
          int firstIndex = (k+1)%Dims;
          int secondIndex = (k+2)%Dims;
	  cross[k] = v1[firstIndex] * v2[secondIndex] - v1[secondIndex] * v2[firstIndex];
        }
        T dist_apod = 0;
        for( int k = 0; k < Dims; k++)
        {
          dist_apod += cross[k]*cross[k];
        }
        dist_apod = sqrt(dist_apod);
        T dist_apod_norm = std::min((T)dist_apod/ref_dist, (T)1.0);
        T apod_weights=0.5*(1.0+cos(M_PI*dist_apod_norm));
	//END APOD WEIGHT


	for(int k = 0; k < output_dimensions; k++)
	{
	  int k_index = k*sample_points*sample_elements;
	  T sample_idx = samples(k_index + tof_idx*elements+j);
	  T sample_idx_plus = samples(k_index + tof_idx_plus*elements+j);
	  T delayed_sample = sample_idx + (sample_idx_plus - sample_idx) * tof_idx_weight;
	  if(j == 0) bf_samples(i + k*points) = 0;
	  bf_samples(i + k*points) += delayed_sample*apod_weights;
	  //if(i == 0 && j == 0 && k == 0) printf("k %d, samp: %e, w: %e, s: %e, sp: %e, t: %d, tp: %d, index: %d, apod: %e\n", k, delayed_sample, tof_idx_weight,  sample_idx, sample_idx_plus, tof_idx*elements+j, tof_idx_plus*elements+j, k_index + tof_idx*elements+j, apod_weights);

	}
	
        delete[] v1;
        delete[] v2;
        delete[] cross;

      }
      delete[] norm_ref;
      delete[] ref_xyz;
      delete[] tof_tmp;
    }
    //for(int i = 0; i < 100; i++)
    //  {
    //	printf("%e\n", tof_pos(i));
    //  }





    std::cout << "Beamform: " << (get_wall_time() - wall_time)*1000.0 << " ms" << std::endl;

  }
};

#define REGISTER_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(						 \
      Name("Beamform").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BeamformOp<type>)

REGISTER_KERNEL(double);
REGISTER_KERNEL(float);

#undef REGISTER_KERNEL




REGISTER_OP("TimeOfFlight")
    .Attr("T: {double, float}")
    .Input("focus: T")
    .Input("center_focus: T")
    .Input("speed_of_sound: T")
    .Input("t_start: T")
    .Input("element_pos: T")
    .Input("point_pos: T")
    .Output("tof_pos: T")
    .Doc(R"doc(
Simple time of flight calc
)doc");

template <typename T>
class TimeOfFlightOp : public OpKernel {
 public:
  explicit TimeOfFlightOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {


    //Read inputs
    const Tensor& focus_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(focus_tensor.shape()),
                errors::InvalidArgument("focus expects a 1-D vector."));
    auto focus = focus_tensor.flat<T>();

    const Tensor& center_focus_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(center_focus_tensor.shape()),
                errors::InvalidArgument("center_focus expects a 1-D vector."));
    auto center_focus = center_focus_tensor.flat<T>();

    const Tensor& speed_of_sound_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(speed_of_sound_tensor.shape()),
                errors::InvalidArgument("speed_of_sound expects a scalar."));
    auto speed_of_sound = speed_of_sound_tensor.flat<T>();

    const Tensor& t_start_tensor = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(t_start_tensor.shape()),
                errors::InvalidArgument("t_start expects a scalar."));
    auto t_start = t_start_tensor.flat<T>();

    const Tensor& element_pos_tensor = context->input(4);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(element_pos_tensor.shape()),
                errors::InvalidArgument("element_pos expects a matrix."));
    auto element_pos = element_pos_tensor.flat<T>();

    const Tensor& point_pos_tensor = context->input(5);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(point_pos_tensor.shape()),
                errors::InvalidArgument("point_pos expects a matrix."));
    auto point_pos = point_pos_tensor.flat<T>();

    double wall_time = get_wall_time();

    // Distance center focus to center
    T tof_cf = 0;
    for (int i = 0; i < focus.size(); i++)
    {
      tof_cf += (focus(i) - center_focus(i)) * (focus(i) - center_focus(i));
    }
    tof_cf = sqrtf(tof_cf);

    //printf("point_pos: %d, %d\n", (int)point_pos_tensor.dim_size(0), (int)point_pos_tensor.dim_size(1));
    // Output a float32 tensor.
    Tensor* tof_pos_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({point_pos_tensor.dim_size(0), 
                                                            element_pos_tensor.dim_size(0)}), &tof_pos_tensor));
    auto tof_pos = tof_pos_tensor->template flat<T>();

    //Loop through all points and get distance to virtual source
    const int P = point_pos_tensor.dim_size(0);
    const int E = element_pos_tensor.dim_size(0);
    const int Dims = point_pos_tensor.dim_size(1);
    auto inv_sof = 1.0/speed_of_sound(0);

    #pragma omp parallel for
    for (int i = 0; i < P; i++) 
    {
      //Calc dist from point to VS(focus)
      T tof_tx = 0;
      for (int k = 0; k < Dims; k++)
      {
        tof_tx += (focus(k) - point_pos(i*Dims+k)) * (focus(k) - point_pos(i*Dims+k));
      }
      tof_tx = sqrtf(tof_tx);
      for (int j = 0; j < E; j++) 
      {
        //Calc dist from point to element
        T tof_rx = 0;
        for (int k = 0; k < Dims; k++)
        {
          tof_rx += (element_pos(j*Dims+k) - point_pos(i*Dims+k)) * (element_pos(j*Dims+k) - point_pos(i*Dims+k));
        }
        tof_rx = sqrtf(tof_rx);
	tof_pos(j + i*E) = (tof_rx+tof_tx+tof_cf)*inv_sof - t_start(0);
      }
    }
    //for(int i = 0; i < 100; i++)
    //  {
    //	printf("%e\n", tof_pos(i));
    //  }

    std::cout << "Time_of_flight: " << (get_wall_time() - wall_time)*1000.0 << " ms" << std::endl;

  }
};

#define REGISTER_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(						 \
      Name("TimeOfFlight").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TimeOfFlightOp<type>)

REGISTER_KERNEL(double);
REGISTER_KERNEL(float);

#undef REGISTER_KERNEL






REGISTER_OP("LinearInterpolation")
    .Attr("T: {double, float}")
    .Input("t_start: T")
    .Input("f_sampling: T")
    .Input("tof_per_bf_point: T")
    .Input("samples: T")
    .Output("tof_samples: T")
    .Doc(R"doc(
Linear interpolation of element samples
)doc");

template <typename T>
class LinearInterpolationOp : public OpKernel {
 public:
  explicit LinearInterpolationOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //Read inputs
    const Tensor& t_start_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(t_start_tensor.shape()),
                errors::InvalidArgument("t_start expects a scalar."));
    auto t_start = t_start_tensor.flat<T>();

    const Tensor& f_sampling_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(f_sampling_tensor.shape()),
                errors::InvalidArgument("f_sampling expects a scalar."));
    auto f_sampling = f_sampling_tensor.flat<T>();

    const Tensor& tof_per_bf_point_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(tof_per_bf_point_tensor.shape()),
                errors::InvalidArgument("tof_per_bf_point expects a matrix."));
    auto tof_per_bf_point = tof_per_bf_point_tensor.flat<T>();

    const Tensor& samples_tensor = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(samples_tensor.shape()),
                errors::InvalidArgument("samples expects a matrix."));
    auto samples = samples_tensor.flat<T>();

    double wall_time = get_wall_time();

    // Output a float32 tensor.
    Tensor* tof_samples_tensor = NULL;
    int output_dimensions = 1;
    if(samples_tensor.dims() > 2)
    {
      output_dimensions = samples_tensor.dim_size(0);
      OP_REQUIRES_OK(context,
                     context->allocate_output(0,  TensorShape({output_dimensions, tof_per_bf_point_tensor.dim_size(0), tof_per_bf_point_tensor.dim_size(1)}), &tof_samples_tensor));
    }
    else
    {
      OP_REQUIRES_OK(context,
                     context->allocate_output(0,  TensorShape({tof_per_bf_point_tensor.dim_size(0), tof_per_bf_point_tensor.dim_size(1)}), &tof_samples_tensor));
    }
    auto tof_samples = tof_samples_tensor->template flat<T>();


    const int dims = output_dimensions;
    const int points = tof_per_bf_point_tensor.dim_size(0);
    const int elements = tof_per_bf_point_tensor.dim_size(1);

    const int sample_points = samples_tensor.dim_size(samples_tensor.dims()-2);
    const int sample_elements = samples_tensor.dim_size(samples_tensor.dims()-1);

    printf("tof: %d, %d, samples: %d, %d\n", points, elements, sample_points, sample_elements);

     
    auto max_index = (int)samples_tensor.dim_size(samples_tensor.dims()-2) - 2;

    #pragma omp parallel for
    for(int k = 0; k < output_dimensions; k++)
    {
      int k_index = k*sample_points*sample_elements;
      for(int i = 0; i < points; i++)
      {
	int elem_index = i*elements+k*elements*points;
	for(int j = 0; j < elements; j++)
	{
	  T tof_idx_weight = (tof_per_bf_point(j + i*elements)-t_start(0))*f_sampling(0);
	  int tof_idx = round(tof_idx_weight);
	  tof_idx_weight = tof_idx_weight - tof_idx;
	  tof_idx = std::max(0, std::min(tof_idx, max_index));
	  int tof_idx_plus = tof_idx + 1;

	  //if(i == 0 && j == 0 && k==0)
	  //printf("tof_tmp: %e, %d, %d, %e, %e\n", tof_per_bf_point(j + i*elements), tof_idx, tof_idx_plus, t_start(0), f_sampling(0));

	  T sample_idx = samples(k_index + tof_idx*elements+j);
	  T sample_idx_plus = samples(k_index + tof_idx_plus*elements+j);
	  //if(i == 0 && j == 0 && k == 0)
	  //  printf("s: %e, sp: %e, %d, %d, %d\n", sample_idx, sample_idx_plus, k_index, tof_idx, tof_idx_plus);
	  //if(i == 0 && j == 0 && k == 0) printf("k %d, w: %e, s: %e, sp: %e, t: %d, tp: %d, index: %d\n", k,  tof_idx_weight,  sample_idx, sample_idx_plus, tof_idx*elements+j, tof_idx_plus*elements+j, k_index + tof_idx*elements+j);
	  tof_samples(j+elem_index) =  sample_idx + (sample_idx_plus - sample_idx) * tof_idx_weight;
	}
      }
    }
    //printf("hurra: %d, %d, samples: %d, %d\n", points, elements, sample_points, sample_elements);

    /*
    #pragma omp parallel for
    for(int i = 0; i < points; i++)
    {

      for(int j = 0; j < elements; j++)
      {
        T tof_idx_weight = (tof_per_bf_point(j + i*elements)-t_start(0))*f_sampling(0);
        int tof_idx = round(tof_idx_weight);
        tof_idx_weight = tof_idx_weight - tof_idx;
        tof_idx = std::max(0, std::min(tof_idx, max_index));

        int tof_idx_plus = tof_idx + 1;
	for(int k = 0; k < output_dimensions; k++)
	{
	  auto k_index = k*sample_points*sample_elements;
	  auto sample_idx = samples(k_index + tof_idx*elements+j);
	  auto sample_idx_plus = samples(k_index + tof_idx_plus*elements+j);
	    
	  tof_samples(j+i*elements+k*elements*points) =  sample_idx + (sample_idx_plus - sample_idx) * tof_idx_weight;
	}
      }
    }
    */
    std::cout << "Linear Interpolation: " << (get_wall_time() - wall_time)*1000.0 << " ms" << std::endl;
  }
};

#define REGISTER_KERNEL(type)                                                   \
  REGISTER_KERNEL_BUILDER(						        \
      Name("LinearInterpolation").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      LinearInterpolationOp<type>)

REGISTER_KERNEL(double);
REGISTER_KERNEL(float);

#undef REGISTER_KERNEL


REGISTER_OP("NoneAndSumApod")
    .Attr("T: {double, float}")    
    .Input("delayed_samples: T")
    .Output("bf_samples: T")
    .Doc(R"doc(
simple sum apodization
)doc");

template <typename T>
class NoneAndSumApodOp : public OpKernel {
 public:
  explicit NoneAndSumApodOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //Read inputs

    const Tensor& delayed_samples_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(delayed_samples_tensor.shape()),
                errors::InvalidArgument("delayed_samples expects a matrix."));
    auto delayed_samples = delayed_samples_tensor.flat<T>();


    double wall_time = get_wall_time();

    // Output a float32 tensor.
    Tensor* bf_samples_tensor = NULL;
    int output_dimensions = 1;
    if(delayed_samples_tensor.dims() > 2)
    {
      output_dimensions = delayed_samples_tensor.dim_size(0);
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({delayed_samples_tensor.dim_size(0), delayed_samples_tensor.dim_size(1)}), &bf_samples_tensor));
    }
    else
    {
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({delayed_samples_tensor.dim_size(0)}), &bf_samples_tensor));
    }
    auto bf_samples = bf_samples_tensor->template flat<T>();

    const int points = delayed_samples_tensor.dim_size(delayed_samples_tensor.dims()-2);
    const int elements = delayed_samples_tensor.dim_size(delayed_samples_tensor.dims()-1);
    #pragma omp parallel for
    for(int k = 0; k < output_dimensions; k++)
    {
      for(int i = 0; i < points; i++)
      {
        T tmp_sum = 0;
        for(int j = 0; j < elements; j++)
        {
          tmp_sum += delayed_samples(k*elements*points+i*elements + j);
        }
        bf_samples(i+k*points) = tmp_sum;
      }
    }

    std::cout << "None_apod: " << (get_wall_time() - wall_time)*1000.0 << " ms" << std::endl;
  }
};


#define REGISTER_KERNEL(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("NoneAndSumApod").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      NoneAndSumApodOp<type>)

REGISTER_KERNEL(double);
REGISTER_KERNEL(float);

#undef REGISTER_KERNEL


//TODO: We only need to calculate the weights once, this could be done in a preprocess step
REGISTER_OP("DynamicAndSumApod")
    .Attr("T: {double, float}")    
    .Input("f_number: T")
    .Input("element_pos: T")
    .Input("bf_point: T")
    .Input("delayed_samples: T")
    .Output("bf_samples: T")
    .Doc(R"doc(
Dynamic sum apodization
)doc");

template <typename T>
class DynamicAndSumApodOp : public OpKernel {
 public:
  explicit DynamicAndSumApodOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //Read inputs
    const Tensor& f_number_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(f_number_tensor.shape()),
                errors::InvalidArgument("f_number expects a scalar."));
    auto f_number = f_number_tensor.flat<T>();

    const Tensor& element_pos_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(element_pos_tensor.shape()),
                errors::InvalidArgument("element_pos expects a matrix."));
    auto element_pos = element_pos_tensor.flat<T>();

    const Tensor& bf_point_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(bf_point_tensor.shape()),
                errors::InvalidArgument("bf_point expects a matrix."));
    auto bf_point = bf_point_tensor.flat<T>();

    const Tensor& delayed_samples_tensor = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(delayed_samples_tensor.shape()),
                errors::InvalidArgument("delayed_samples expects a matrix."));
    auto delayed_samples = delayed_samples_tensor.flat<T>();

    double wall_time = get_wall_time();

    // Output a float32 tensor.
    Tensor* bf_samples_tensor = NULL;
    int output_dimensions = 1;
    if(delayed_samples_tensor.dims() > 2)
    {
      output_dimensions = delayed_samples_tensor.dim_size(0);
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({delayed_samples_tensor.dim_size(0), delayed_samples_tensor.dim_size(1)}), &bf_samples_tensor));
    }
    else
    {
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({delayed_samples_tensor.dim_size(0)}), &bf_samples_tensor));
    }
    auto bf_samples = bf_samples_tensor->template flat<T>();


    const int points = bf_point_tensor.dim_size(0);
    const int elements = element_pos_tensor.dim_size(0);
    const int Dims = element_pos_tensor.dim_size(1);


    #pragma omp parallel for
    for(int i = 0; i < points; i++)
    {
      T *ref_xyz = new T[Dims];
      T ref_dist = 1000000;
      for(int j = 0; j < elements; j++)
      {
        //Calc dist from point to element
        T tof_rx = 0;
        for (int k = 0; k < Dims; k++)
        {
          tof_rx += (element_pos(j*Dims+k) - bf_point(i*Dims+k)) * (element_pos(j*Dims+k) - bf_point(i*Dims+k));
        }
        tof_rx = sqrtf(tof_rx);
        //Save closest distance to element and vector from element to bf_point
        if(tof_rx < ref_dist)
        { 
          ref_dist = tof_rx;
          for (int k = 0; k < Dims; k++)
          {
            ref_xyz[k] = element_pos(j*Dims+k) - bf_point(i*Dims+k);
          }
        }

      }
      T* norm_ref = new T[Dims];
      for (int k = 0; k < Dims; k++)
      {
	norm_ref[k] = bf_point(i*Dims+k) + ref_xyz[k] / ref_dist;
      }

      //Calc distance to line for all elements
      ref_dist = ref_dist/f_number(0) * 0.5;
      //if( i == 0)
      //printf("i: %d, ref_dist: %f\n", i, ref_dist);

      
      for(int j = 0; j < elements; j++)
      {
        T* v1 = new T[Dims];
        T* v2 = new T[Dims];
        for (int k = 0; k < Dims; k++)
        {
          v1[k] = element_pos(j*Dims+k) - norm_ref[k];
          v2[k] = element_pos(j*Dims+k) - bf_point(i*Dims+k);
        }
        T* cross = new T[Dims];
	for( int k = 0; k < Dims; k++)
        {
          int firstIndex = (k+1)%Dims;
          int secondIndex = (k+2)%Dims;
	  cross[k] = v1[firstIndex] * v2[secondIndex] - v1[secondIndex] * v2[firstIndex];
        }
        T dist_apod = 0;
        for( int k = 0; k < Dims; k++)
        {
          dist_apod += cross[k]*cross[k];
        }
        dist_apod = sqrt(dist_apod);
        T dist_apod_norm = std::min((T)dist_apod/ref_dist, (T)1.0);
        T apod_weights=0.5*(1.0+cos(M_PI*dist_apod_norm));
	//printf("dist_apod_norm: %d, %d, %f, apod_weights: %f\n", i, j, dist_apod_norm, apod_weights);
	for(int k = 0; k < output_dimensions; k++)
	{
	  if(j == 0) bf_samples(i + k*points) = 0;
	  bf_samples(i + k*points) += delayed_samples(k*points*elements + i*elements+j)*apod_weights;
	  //if(i == 0 && j == 0 && k == 0) printf("k %d, samp: %e, index: %d, apod: %e\n", k, delayed_samples(k*points*elements + i*elements+j), k*points*elements + i*elements+j, apod_weights);

	}
        delete[] v1;
        delete[] v2;
        delete[] cross;
      }

      delete[] norm_ref;
      delete[] ref_xyz;
    }
    //for(int i = 0; i < 100; i++)
    //  printf("%e\n", bf_samples(i));

    std::cout << "Dynamic_apod: " << (get_wall_time() - wall_time)*1000.0 << " ms" << std::endl;
  }
};

#define REGISTER_KERNEL(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("DynamicAndSumApod").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      DynamicAndSumApodOp<type>)

REGISTER_KERNEL(double);
REGISTER_KERNEL(float);

#undef REGISTER_KERNEL


REGISTER_OP("EchoCancel")
    .Attr("T: {float, double}")    
    .Input("samples: T")
    .Output("echo_samples: T")
    .Doc(R"doc(
Simple mean-value echo cancelling
)doc");

template <typename T>
class EchoCancelOp : public OpKernel {
 public:
  explicit EchoCancelOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //Read inputs
    const Tensor& samples_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(samples_tensor.shape()),
                errors::InvalidArgument("samples expects a matrix."));
    auto samples = samples_tensor.flat<T>();

    double wall_time = get_wall_time();

    // Output a tensor.
    Tensor* echo_samples_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, samples_tensor.shape(), &echo_samples_tensor));
    auto echo_samples = echo_samples_tensor->template flat<T>();
    const int batch_size = samples_tensor.dim_size(0);
    const int points = samples_tensor.dim_size(1);

    T *mean_values = new T[points];    
    //Find mean of all points in batch
    #pragma omp parallel for
    for(int i = 0; i < points; i++)
    {
      T mean_value = 0;
      for(int j = 0; j < batch_size; j++)
      {
        mean_value += samples(j*points + i);
      }
      mean_values[i] = mean_value/batch_size;
    }

    for(int i = 0; i < points; i++)
    {
      for(int j = 0; j < batch_size; j++)
      {
        echo_samples(j * points + i) = samples(j * points + i) - mean_values[i];
      }
    }
    delete[] mean_values;

    std::cout << "Echo_cancel: " << (get_wall_time() - wall_time)*1000.0 << " ms" << std::endl;
  }
};

#define REGISTER_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("EchoCancel").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      EchoCancelOp<type>)

REGISTER_KERNEL(double);
REGISTER_KERNEL(float);

#undef REGISTER_KERNEL


REGISTER_OP("EchoCancelThreshold")
    .Attr("T: {float, double}")    
    .Input("samples: T")
    .Input("fourier_threshold: T")
    .Input("tukey_ec: T")
    .Input("tukey_ec_freq: T")
    .Input("percentage: T")
    .Output("echo_samples: T")
    .Doc(R"doc(
Echo cancelling with fourier thresholding
)doc");

template <typename T>
class EchoCancelThresholdOp : public OpKernel {
 public:
  explicit EchoCancelThresholdOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //Read inputs
    const Tensor& samples_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(samples_tensor.shape()),
                errors::InvalidArgument("samples expects a matrix."));
    auto samples = samples_tensor.flat<T>();

    const Tensor& fourier_threshold_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(fourier_threshold_tensor.shape()),
                errors::InvalidArgument("fourier threshold expects a Scalar."));
    auto fourier_threshold = fourier_threshold_tensor.flat<T>();

    const Tensor& tukey_ec_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(tukey_ec_tensor.shape()),
                errors::InvalidArgument("tukey_ec expects a Vector."));
    auto tukey_ec = tukey_ec_tensor.flat<T>();

    const Tensor& tukey_ec_freq_tensor = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(tukey_ec_freq_tensor.shape()),
                errors::InvalidArgument("tukey_ec_freq expects a Vector."));
    auto tukey_ec_freq = tukey_ec_freq_tensor.flat<T>();

    const Tensor& percentage_tensor = context->input(4);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(percentage_tensor.shape()),
                errors::InvalidArgument("percentage expects a Scalar."));
    auto percentage = percentage_tensor.flat<T>();


    double wall_time = get_wall_time();

    // Output a tensor.
    Tensor* echo_samples_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, samples_tensor.shape(), &echo_samples_tensor));
    auto echo_samples = echo_samples_tensor->template flat<T>();



    const int batch_size = samples_tensor.dim_size(0);
    const int points = samples_tensor.dim_size(1);


    T *samples_reorder = new T[points*batch_size];
    #pragma omp parallel for
    for(int i = 0; i < points; i++) {
      for(int j = 0; j < batch_size; j++) {
        samples_reorder[j + i*batch_size] = samples(j*points + i);
      }
    }


    //printf("batch_size: %d, points: %d\n", batch_size, points);


    int fft_len = batch_size;

    fftw_complex *fft_signal, *fft_out, *fft_result;
    fftw_plan pa, pb;

    fft_signal = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);
    fft_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);
    fft_result = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);


    pa = fftw_plan_dft_1d(fft_len, fft_signal, fft_out, FFTW_FORWARD, FFTW_MEASURE);//FFTW_EXHAUSTIVE);
    pb = fftw_plan_dft_1d(fft_len, fft_out, fft_result, FFTW_BACKWARD, FFTW_MEASURE);//FFTW_EXHAUSTIVE);


    std::complex<double> scale = 1.0/fft_len;

    #pragma omp parallel for
    for(int i = 0; i < points; i++)
    { 
      
      fftw_complex *fft_signal_p, *fft_out_p, *fft_result_p;
      fft_signal_p = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);
      fft_out_p = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);
      fft_result_p = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);

      memset(fft_signal_p, 0, sizeof(fftw_complex) * fft_len);

      for(int j = 0; j < batch_size; j++)
      {
	//Read input from samples and apply tukey_ec window
        fft_signal_p[j][0] = samples_reorder[i*batch_size + j] * tukey_ec(j);
      }
 
      //Calculate first DFT 
      fftw_execute_dft(pa, fft_signal_p, fft_out_p);
	  
      std::complex<double> out1;
      for(int j = 0; j < fft_len; j++)
      {
	out1.real(fft_out_p[j][0]);
	out1.imag(fft_out_p[j][1]);
	//if(i < 2) printf("%d, %e, %e, %e, %e, %e, %e\n", j, samples(j), samples_reorder[i*batch_size + j], fft_signal_p[j][0], fft_signal_p[j][1], out1.real(), out1.imag());
	double rval = std::abs(out1);
	//if(i<2)printf("%d, rval: %e %e\n", j, rval, fourier_threshold(0));
	if (rval > fourier_threshold(0))
	{
	  rval = fourier_threshold(0) * percentage(0);
	}

	std::complex<double> out2;
	out2.real(rval * tukey_ec_freq(j));
	out1 = out2 * std::exp(std::complex<double>(0, std::atan2(out1.imag(), out1.real()))) * scale;
	//if(i<2)printf("%d, %e, %e, %e\n", j, rval, out1.real(), out1.imag());

	fft_out_p[j][0] = out1.real();
	fft_out_p[j][1] = out1.imag();
      }
	  
      //Inverse DFT
      fftw_execute_dft(pb, fft_out_p, fft_result_p);

      for(int j = 0; j < batch_size; j++)
      {
	echo_samples(j * points + i) = fft_result_p[j][0];
	//if(i < 2) printf("%d, %e, %e, %e, %e\n", j, samples(j), samples_reorder[i*batch_size + j], fft_result_p[j][0], fft_result_p[j][1]);
      }


      fftw_free(fft_signal_p); 
      fftw_free(fft_out_p);
      fftw_free(fft_result_p);
      
    }


  

    delete[] samples_reorder;

    fftw_destroy_plan(pa);
    fftw_destroy_plan(pb);
    fftw_free(fft_signal); 
    fftw_free(fft_out);
    fftw_free(fft_result);


    fftw_cleanup();


    std::cout << "Echo_cancel (fourier thresholding): " << (get_wall_time() - wall_time)*1000.0 << " ms" << std::endl;
  }
};

#define REGISTER_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("EchoCancelThreshold").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      EchoCancelThresholdOp<type>)

REGISTER_KERNEL(double);
REGISTER_KERNEL(float);

#undef REGISTER_KERNEL



REGISTER_OP("CartToPolar")
    .Attr("T: {float, double}")    
    .Input("input_point: T")
    .Input("lambda: T")
    .Input("line_length: T")
    .Input("d_line: T")
    .Input("bf_points: T")
    .Input("bf_grid: int32")
    .Input("angles: T")
    .Input("samples: T")
    .Input("all_angles: bool")
    .Output("polar_samples: T")
    .Doc(R"doc(
Cartesian to polar interpolation on grid
)doc");


//TODO: ADD Y dimension, find better solution when each point corresponds to one angle instead of testing all angles
//Currently uses a boolean
template <typename T>
class CartToPolarOp : public OpKernel {
 public:
  explicit CartToPolarOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //Read inputs
    //TODO: Could be both one point or matrix - need to fix
    const Tensor& input_points_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_points_tensor.shape()) || TensorShapeUtils::IsVector(input_points_tensor.shape()),
                errors::InvalidArgument("input_points expects a vector or matrix."));
    auto input_points = input_points_tensor.flat<T>();

    const Tensor& lambda_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(lambda_tensor.shape()),
                errors::InvalidArgument("lambda expects a scalar."));
    auto lambda = lambda_tensor.flat<double>();

    const Tensor& line_length_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(line_length_tensor.shape()),
                errors::InvalidArgument("line_length expects a scalar."));
    auto line_length = line_length_tensor.flat<T>();

    const Tensor& d_line_tensor = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(d_line_tensor.shape()),
                errors::InvalidArgument("d_line expects a scalar."));
    auto d_line = d_line_tensor.flat<T>();

 
    const Tensor& bf_points_tensor = context->input(4);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(bf_points_tensor.shape()),
                errors::InvalidArgument("bf_points expects a matrix."));
    auto bf_points = bf_points_tensor.flat<T>();

    const Tensor& bf_grid_tensor = context->input(5);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(bf_grid_tensor.shape()),
                errors::InvalidArgument("bf_grid expects a vector."));
    auto bf_grid = bf_grid_tensor.flat<int>();

    //TODO: Fix that angles can be both one angle and multiple
    const Tensor& angles_tensor = context->input(6);
    //printf("angles: %d\n", (int)angles_tensor.dim_size(0));
    //OP_REQUIRES(context, TensorShapeUtils::IsVector(angles_tensor.shape()) || TensorShapeUtils::IsScalar(angles_tensor.shape()),
    //            errors::InvalidArgument("angles expects a scalar or vector."));
    auto angles = angles_tensor.flat<T>();

    const Tensor& samples_tensor = context->input(7);
    //printf("samples: %d, %d\n", (int)samples_tensor.dim_size(0), (int)samples_tensor.dim_size(1));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(samples_tensor.shape()),
                errors::InvalidArgument("samples expects a matrix."));
    auto samples = samples_tensor.flat<T>();

    const Tensor& all_angles_tensor = context->input(8);
    //printf("angles: %d\n", (int)angles_tensor.dim_size(0));
    //OP_REQUIRES(context, TensorShapeUtils::IsVector(angles_tensor.shape()) || TensorShapeUtils::IsScalar(angles_tensor.shape()),
    //            errors::InvalidArgument("angles expects a scalar or vector."));
    auto all_angles = all_angles_tensor.flat<bool>();

    double wall_time = get_wall_time();

    int r_val_length = (int)(line_length(0)/d_line(0));
    double *r_values = new double[r_val_length];
    for(int i = 0; i < r_val_length; i++)
    {
      r_values[i] = (-line_length(0)/2.0 + i*d_line(0))*lambda(0);
    }


    int ip_size = 1;
    int angles_pr_point = angles_tensor.dim_size(0);
    int point_offset = 0;
    if(!all_angles(0))
    {
      point_offset = 1;
      angles_pr_point = 1;
    }

    // Output a float32 tensor.
    Tensor* polar_samples_tensor = NULL;
    if(input_points_tensor.dims() > 1)
    {
      ip_size = input_points_tensor.dim_size(0);
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({input_points_tensor.dim_size(0), samples_tensor.dim_size(0), angles_pr_point, r_val_length}), 
					      &polar_samples_tensor));
    }
    else
    {
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({samples_tensor.dim_size(0), angles_pr_point, r_val_length}), 
					      &polar_samples_tensor));

    }
    auto polar_samples = polar_samples_tensor->template flat<double>();

    int total_angles = angles_tensor.dim_size(0);
    double* TH = new double[total_angles];
    #pragma omp parallel for
    for(int i = 0; i < total_angles; i++)
    {
      TH[i] = (angles(i)+90.0)*M_PI/180.0;
      //printf("TH: %d, %.15f\n", i, TH[i]);
    }


    int g_x = (int)bf_grid(0);
    int g_y = (int)bf_grid(1);
    int g_z = (int)bf_grid(2);
    int g_dim = bf_grid_tensor.dim_size(0);

    int batch_size = samples_tensor.dim_size(0);
    int no_of_points = samples_tensor.dim_size(1);

    
    //double* pol2cart_x = new double[r_val_length*angles_pr_point]; 
    //double* pol2cart_z = new double[r_val_length*angles_pr_point]; 
    double* pol2cart_x = new double[ip_size*r_val_length*angles_pr_point]; 
    double* pol2cart_z = new double[ip_size*r_val_length*angles_pr_point]; 

    #pragma omp parallel for
    for(int ip = 0; ip < ip_size; ip++)
    {
      int angle_index = total_angles;
      int start_index = 0;
      if(!all_angles(0))
	{
	  start_index = ip;
	  angle_index = ip+1;
	}

      //Calc line to sample for input point 

      for(int k = 0, i = start_index; i < angle_index; i++, k++)
      {
        for(int j = 0; j < r_val_length; j++)
        {
	  pol2cart_x[j+k*r_val_length+ip*angles_pr_point*r_val_length] = input_points(ip*3 + 0) + r_values[j]*cos(TH[i]);
	  pol2cart_z[j+k*r_val_length+ip*angles_pr_point*r_val_length] = input_points(ip*3 + 2) + r_values[j]*sin(TH[i]);
        }
      }
    }

    #pragma omp parallel for
    for(int ip = 0; ip < ip_size; ip++)
    {
      int angle_index = total_angles;
      int start_index = 0;
      if(!all_angles(0))
	{
	  start_index = ip;
	  angle_index = ip+1;
	}

      //#pragma omp parallel for
      for(int l = 0, i = start_index; i < angle_index; i++, l++)
      {
	//#pragma omp parallel for
        for(int j = 0; j < r_val_length; j++)
        {
	  int x_index = 0;
	  double weight_x = 0;
	  int y_index = 0;
          double weight_y = 0;
	  int z_index = 0;
	  double weight_z = 0;
	  for(int k = 0; k < g_x-1; k++)
	  {
	    double pol_x = pol2cart_x[j+l*r_val_length+ip*angles_pr_point*r_val_length];
	    double bf_p1 = bf_points(k*g_dim);
            double bf_p2 = bf_points((k+1)*g_dim); //dim_size probably = 3
	    if(bf_p1 <= pol_x && pol_x <= bf_p2)
	    {
	      weight_x =  (pol_x -  bf_p1) / (bf_p2 -  bf_p1);
	      x_index = k;
	      break;
	    }
	  }
	  /*for(int k = 0; k < g_y-1; k++)
	  {
	    auto pol_y = pol2cart_y[j+i*r_val_length];
	    if(bf_points[k] <= pol_x && pol_x <= bf_points[k+1])
	    {
	      y_index = k;
	      break;
	    }
	  }
	  */
	  for(int k = 0; k < g_z-1; k++)
	  {
	    double pol_z = pol2cart_z[j+l*r_val_length+ip*angles_pr_point*r_val_length];
	    double bf_p1 = bf_points(g_dim*k + (g_dim-1));
            double bf_p2 = bf_points(g_dim*(k+1) + (g_dim-1)); //dim_size probably = 3
	    if(bf_p1 <= pol_z && pol_z <= bf_p2)
	    {
	      weight_z =  (pol_z - bf_p1) / (bf_p2 -  bf_p1);
	      z_index = k;
	      break;
	    }
	  }
	  
	  for(int k = 0; k < batch_size; k++)
	  {
	    double v0 = samples((x_index+1) + z_index*g_x + k*no_of_points) * weight_x + samples((x_index) + z_index*g_x + k*no_of_points)*(1.0 - weight_x);
	    double v1 = samples((x_index+1) + (z_index+1)*g_x + k*no_of_points) * weight_x + samples((x_index) + (z_index+1)*g_x + k*no_of_points)*(1.0 - weight_x);
	    //Output to polar samples
	    int pol_index = j + l*r_val_length + k*r_val_length*angles_pr_point + ip*r_val_length*angles_pr_point*batch_size;
	    polar_samples(pol_index) = v1 * weight_z + v0 * (1.0-weight_z);
	    //	  printf("samples: %d, %.15f\n", k, 
	  }
        }
      }
    }

    /*


    //#pragma omp parallel for
    for(int ip = 0; ip < ip_size; ip++)
    {
      int angle_index = total_angles;
      int start_index = 0;
      if(!all_angles(0))
	{
	  start_index = ip;
	  angle_index = ip+1;
	}

      //Calc line to sample for input point 

      for(int k = 0, i = start_index; i < angle_index; i++, k++)
      {
        for(int j = 0; j < r_val_length; j++)
        {
	  pol2cart_x[j+k*r_val_length] = input_points(ip*3 + 0) + r_values[j]*cos(TH[i]);
	  pol2cart_z[j+k*r_val_length] = input_points(ip*3 + 2) + r_values[j]*sin(TH[i]);
        }
      }

      //#pragma omp parallel for
      for(int l = 0, i = start_index; i < angle_index; i++, l++)
      {
	//#pragma omp parallel for
        for(int j = 0; j < r_val_length; j++)
        {
	  int x_index = 0;
	  double weight_x = 0;
	  int y_index = 0;
          double weight_y = 0;
	  int z_index = 0;
	  double weight_z = 0;
	  for(int k = 0; k < g_x-1; k++)
	  {
	    double pol_x = pol2cart_x[j+l*r_val_length];
	    double bf_p1 = bf_points(k*g_dim);
            double bf_p2 = bf_points((k+1)*g_dim); //dim_size probably = 3
	    if(bf_p1 <= pol_x && pol_x <= bf_p2)
	    {
	      weight_x =  (pol_x -  bf_p1) / (bf_p2 -  bf_p1);
	      x_index = k;
	      break;
	    }
	  }
	  //for(int k = 0; k < g_y-1; k++)
	  //{
	  // auto pol_y = pol2cart_y[j+i*r_val_length];
	  //  if(bf_points[k] <= pol_x && pol_x <= bf_points[k+1])
	  //  {
	  //    y_index = k;
	  //    break;
	  //  }
	  //}
	  
	  for(int k = 0; k < g_z-1; k++)
	  {
	    double pol_z = pol2cart_z[j+l*r_val_length];
	    double bf_p1 = bf_points(g_dim*k + (g_dim-1));
            double bf_p2 = bf_points(g_dim*(k+1) + (g_dim-1)); //dim_size probably = 3
	    if(bf_p1 <= pol_z && pol_z <= bf_p2)
	    {
	      weight_z =  (pol_z - bf_p1) / (bf_p2 -  bf_p1);
	      z_index = k;
	      break;
	    }
	  }

	  for(int k = 0; k < batch_size; k++)
	  {
	    double v0 = samples((x_index+1) + z_index*g_x + k*no_of_points) * weight_x + samples((x_index) + z_index*g_x + k*no_of_points)*(1.0 - weight_x);
	    double v1 = samples((x_index+1) + (z_index+1)*g_x + k*no_of_points) * weight_x + samples((x_index) + (z_index+1)*g_x + k*no_of_points)*(1.0 - weight_x);
	    //Output to polar samples
	    int pol_index = j + l*r_val_length + k*r_val_length*angles_pr_point + ip*r_val_length*angles_pr_point*batch_size;
	    polar_samples(pol_index) = v1 * weight_z + v0 * (1.0-weight_z);
	    //	  printf("samples: %d, %.15f\n", k, 
	  }
        }
      }
    }
    */
    std::cout << "Cart_to_polar: " << (get_wall_time() - wall_time)*1000.0 << " ms" << std::endl;

    delete[] pol2cart_x;    
    delete[] pol2cart_z;

    delete[] r_values;
    delete[] TH;

  }
};

#define REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("CartToPolar").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      CartToPolarOp<type>)

REGISTER_KERNEL(double);
REGISTER_KERNEL(float);

#undef REGISTER_KERNEL


REGISTER_OP("ParabolicXCorr")
    .Attr("T: {float, double}")    
    .Input("lambda: T")
    .Input("t_prf_eff: T")
    .Input("line_length: T")
    .Input("d_line: T")
    .Input("samples: T")
    .Input("tukey: T")
    .Output("out_samples: T")
    .Doc(R"doc(
Cartesian to polar interpolation on grid
)doc");

template <typename T>
class ParabolicXCorrOp : public OpKernel {
 public:
  explicit ParabolicXCorrOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //Read inputs
    const Tensor& lambda_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(lambda_tensor.shape()),
                errors::InvalidArgument("lambda expects a scalar."));
    auto lambda = lambda_tensor.flat<T>();

    const Tensor& t_prf_eff_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(t_prf_eff_tensor.shape()),
                errors::InvalidArgument("t_prf_eff expects a scalar."));
    auto t_prf_eff = t_prf_eff_tensor.flat<T>();

    const Tensor& line_length_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(line_length_tensor.shape()),
                errors::InvalidArgument("line_length expects a scalar."));
    auto line_length = line_length_tensor.flat<T>();

    const Tensor& d_line_tensor = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(d_line_tensor.shape()),
                errors::InvalidArgument("d_line expects a scalar."));
    auto d_line = d_line_tensor.flat<T>();

    const Tensor& samples_tensor = context->input(4);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(samples_tensor.shape()),
                errors::InvalidArgument("samples expects a matrix or higher."));
    auto samples = samples_tensor.flat<T>();

    const Tensor& tukey_tensor = context->input(5);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(tukey_tensor.shape()),
                errors::InvalidArgument("tukey expects a vector."));
    auto tukey = tukey_tensor.flat<T>();

    double wall_time = get_wall_time();

    int input_points = 1;
    int sample_dims = samples_tensor.dims();
    // Output a tensor.

    //printf("Samples_tensor: %d, %d, %d, %d\n", samples_tensor.dims(), samples_tensor.dim_size(0), samples_tensor.dim_size(1), samples_tensor.dim_size(2));
    Tensor* v_flow_tensor = NULL;
    if(sample_dims > 3)
    {
      input_points = samples_tensor.dim_size(0);
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({samples_tensor.dim_size(0), samples_tensor.dim_size(2)}), 
					      &v_flow_tensor));
    }
    else
    {
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({samples_tensor.dim_size(1)}), 
					      &v_flow_tensor));
    }
    auto v_flow = v_flow_tensor->template flat<T>();

    int lag_limit = floor(line_length(0)/d_line(0)-1);


    int batches = samples_tensor.dim_size(sample_dims - 3);
    int angles = samples_tensor.dim_size(sample_dims - 2);
    int positions = samples_tensor.dim_size(sample_dims - 1);

    int fft_len = positions*2-1;
    
    T *row_angle_sums = new T[fft_len*angles];

    int *angles_maximum = new int[angles];

    int center_lag = (int)((fft_len-1)/2+1);

    
    fftw_complex *fft_signal_a, *fft_signal_b, *fft_out_a, *fft_out_b, *fft_out, *fft_result;
    fftw_plan pa, pb, pc;

    //fftw_init_threads();
    //fftw_plan_with_nthreads(4);
   // int wisdom = fftw_import_wisdom_from_filename("wisdom.txt");

    

    fft_signal_a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);
    fft_out_a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);
    fft_signal_b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);
    fft_out_b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);
    fft_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);
    fft_result = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);


   // if(wisdom != 0)
    //{
      //pa = fftw_plan_dft_1d(fft_len, fft_signal_a, fft_out_a, FFTW_FORWARD, FFTW_WISDOM_ONLY);
      //pb = fftw_plan_dft_1d(fft_len, fft_signal_b, fft_out_b, FFTW_FORWARD, FFTW_WISDOM_ONLY);
      //pc = fftw_plan_dft_1d(fft_len, fft_out, fft_result, FFTW_BACKWARD, FFTW_WISDOM_ONLY);
    //}
    //else
    //{
      pa = fftw_plan_dft_1d(fft_len, fft_signal_a, fft_out_a, FFTW_FORWARD, FFTW_MEASURE);//FFTW_EXHAUSTIVE);
      pb = fftw_plan_dft_1d(fft_len, fft_signal_b, fft_out_b, FFTW_FORWARD, FFTW_MEASURE);//FFTW_EXHAUSTIVE);
      pc = fftw_plan_dft_1d(fft_len, fft_out, fft_result, FFTW_BACKWARD, FFTW_MEASURE);//FFTW_EXHAUSTIVE);
      //fftw_export_wisdom_to_filename("wisdom.txt");
      //printf("export wisdom\n");
    //}

    std::complex<double> scale = 1.0/fft_len;
    int lag_lower_bound = center_lag - lag_limit;
    int lag_upper_bound = fft_len - (lag_lower_bound);

    for(int ip = 0; ip < input_points; ip++)
    { 
      memset(row_angle_sums, 0, fft_len*angles*sizeof(T)); 
      memset(angles_maximum, 0, angles*sizeof(int));   
      
      #pragma omp parallel for
      for(int k = 0; k < angles; k++)
      {

        fftw_complex *fft_signal_aa, *fft_signal_bb, *fft_out_aa, *fft_out_bb, *fft_out_x, *fft_result_x;
        fft_signal_aa = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);
        fft_out_aa = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);
	fft_signal_bb = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);
	fft_out_bb = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);
	fft_out_x = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);
	fft_result_x = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_len);
	
        for(int i = 0; i < batches-1; i++)
        {	
          memset(fft_signal_aa, 0, sizeof(fftw_complex) * fft_len);
	  memset(fft_signal_bb, 0, sizeof(fftw_complex) * fft_len);

	  //Read input from samples and apply tukey window
	  for(int j = 0; j < positions; j++)
	  {
	    //Input signal is shifted here corresponds to fftwshift in matlab
            fft_signal_aa[j+(positions-1)][0] = samples(ip*batches*positions*angles + i*positions*angles + k*positions + j) * tukey(j);
	    fft_signal_bb[j][0] = samples(ip*batches*positions*angles + (i+1)*positions*angles + k*positions + j) * tukey(j);
	  }

	  //Calculate first two DFT and then run c = a.*conj(b)
	  fftw_execute_dft(pa, fft_signal_aa, fft_out_aa);
	  fftw_execute_dft(pb, fft_signal_bb, fft_out_bb);

	  
	  std::complex<double> out1;
	  std::complex<double> out2;
	  for(int j = 0; j < fft_len; j++)
	  {
            out1.real(fft_out_aa[j][0]);
	    out1.imag(fft_out_aa[j][1]);
	    out2.real(fft_out_bb[j][0]);
	    out2.imag(fft_out_bb[j][1]);
	    out1 = out1 * std::conj(out2) * scale;
	    fft_out_x[j][0] = out1.real();
	    fft_out_x[j][1] = out1.imag();
	  }
	  
	  //Inverse DFT
	  fftw_execute_dft(pc, fft_out_x, fft_result_x);

	  
	  //Calc mean value
	  for(int j = 0; j < fft_len; j++)
	  {
	    if(j > lag_lower_bound && j < lag_upper_bound)
	      {
		row_angle_sums[k*fft_len + j] += fft_result_x[j][0]/(batches-1);

	      }

	  }
	  


        }
	fftw_free(fft_signal_aa); 
	fftw_free(fft_out_aa);
	fftw_free(fft_signal_bb); 
	fftw_free(fft_out_bb);
	fftw_free(fft_out_x);
	fftw_free(fft_result_x);

      }

      #pragma omp parallel for
      for(int i = 0; i < angles; i++)
      {
        T angle_maximum = row_angle_sums[i*fft_len];
        for(int j = 0; j < fft_len; j++)
        {
  	  if(row_angle_sums[i*fft_len + j] > angle_maximum)
	  {
	    angle_maximum = row_angle_sums[i*fft_len + j];
	    angles_maximum[i] = j;
	  }

        }
      }
    
      #pragma omp parallel for
      for(int i = 0; i < angles; i++)
      {
        int index = i*fft_len+angles_maximum[i];
        T index_val = row_angle_sums[index];
        T indexp_val = row_angle_sums[index+1];
        T indexm_val = row_angle_sums[index-1];
        T interp_lag = angles_maximum[i] - (indexp_val - indexm_val) / (2*(indexp_val - 2*index_val + indexm_val));
        v_flow(ip*angles + i) = ((interp_lag - (center_lag-1))*d_line(0)*lambda(0)/t_prf_eff(0));
      }
    }

    fftw_destroy_plan(pa);
    fftw_destroy_plan(pb);
    fftw_destroy_plan(pc);
    fftw_free(fft_signal_a); 
    fftw_free(fft_out_a);
    fftw_free(fft_signal_b); 
    fftw_free(fft_out_b);
    fftw_free(fft_out);
    fftw_free(fft_result);

    fftw_cleanup();


    std::cout << "ParabolicXCorr: " << (get_wall_time() - wall_time)*1000.0 << " ms" << std::endl;    

    delete[] angles_maximum;
    delete[] row_angle_sums;

  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ParabolicXCorr").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ParabolicXCorrOp<type>)

REGISTER_KERNEL(double);
REGISTER_KERNEL(float);

#undef REGISTER_KERNEL


REGISTER_OP("MCD3D")
    .Attr("T: {float, double}")    
    .Input("samples: T")
    .Input("angles_theta: T")
    .Input("angles_phi: T")
    .Output("angle: T")
    .Doc(R"doc(
Compare distances between velocity/angles and return minimum distance angle
)doc");


//TODO: ADD Y dimension
template <typename T>
class MCD3DOp : public OpKernel {
 public:
  explicit MCD3DOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override 
{
    //Read inputs
    const Tensor& samples_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(samples_tensor.shape()),
                errors::InvalidArgument("samples expects a matrix or higher."));
    auto samples = samples_tensor.flat<T>();

    const Tensor& angles_theta_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(angles_theta_tensor.shape()),
                errors::InvalidArgument("angles_theta expects a vector."));
    auto angles_theta = angles_theta_tensor.flat_inner_dims<T>();

    const Tensor& angles_phi_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(angles_phi_tensor.shape()),
                errors::InvalidArgument("angles_phi expects a vector."));
    auto angles_phi = angles_phi_tensor.flat_inner_dims<T>();


    double wall_time = get_wall_time();

    int input_points = 1;
    int samples_dims = samples_tensor.dims();
    int angles_theta_size = angles_theta_tensor.dim_size(0);
    int angles_phi_size = angles_phi_tensor.dim_size(0);

    // Output a tensor.
    Tensor* angle_tensor = NULL;
    if(samples_tensor.dims() > 2)
    {
      input_points = samples_tensor.dim_size(1);
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({samples_tensor.dim_size(1), 2}), 
					      &angle_tensor));
    }
    else
    {
          OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({2}), 
					      &angle_tensor));
    }
    auto angle = angle_tensor->template flat<T>();



    int emission_types = samples_tensor.dim_size(0);
    int velocities = samples_tensor.dim_size(samples_dims-1);

    

    for(int ip = 0; ip < input_points; ip++)
    {
      T* vel_diff = new T[velocities];
      memset(vel_diff, 0, sizeof(T)*velocities);
    

      for(int i = 0; i < emission_types; i++)
      {
	for(int j = i+1; j < emission_types; j++)
	{
	  for(int k = 0; k < velocities; k++)
	  {
	    T first_sample = samples(k + i*velocities*input_points + ip*velocities);
	    T second_sample = samples(k + j*velocities*input_points + ip*velocities);
	    T diff = first_sample - second_sample;
	    diff = std::abs(diff / std::abs(std::min(first_sample, second_sample)));
	    vel_diff[k] += diff;
	  }
	}
      }

      T min_dist = 100000000;

      for(int i = 0; i < velocities; i++)
      {
	//printf("vel_diff: %.10f\n", vel_diff[i]);
	if(min_dist > vel_diff[i])
	{
	  min_dist = vel_diff[i];
	  angle(2*ip) = angles_theta(i % angles_theta_size);
	  angle(2*ip+1) = angles_phi((int)(i / angles_theta_size));
        }
      }
      delete[] vel_diff;
    }
    std::cout << "MCD3D: " << (get_wall_time() - wall_time)*1000.0 << " ms" << std::endl;
  }
};

#define REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MCD3D").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      MCD3DOp<type>)

REGISTER_KERNEL(double);
REGISTER_KERNEL(float);

#undef REGISTER_KERNEL


REGISTER_OP("AddOne")
    .Input("input: int32")
    .Output("output: int32")
    .Doc(R"doc(
Adds 1 to all elements of the tensor.
output: A Tensor.
  output = input + 1
)doc");

void AddOneKernelLauncher(const int* in, const int N, int* out);

class AddOneOp : public OpKernel {
 public:

  explicit AddOneOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    // Call the cuda kernel launcher
    //AddOneKernelLauncher(input.data(), N, output.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_CPU), AddOneOp);


typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


REGISTER_OP("Beamforming")
    .Attr("T: {double, float}")
    .Input("focus: T")
    .Input("center_focus: T")
    .Input("speed_of_sound: T")
    .Input("t_start: T")
    .Input("t_start_data: T")
    .Input("f_sampling: T")
    .Input("f_number: T")
    .Input("element_pos: T")
    .Input("point_pos: T")
    .Input("samples: T")
    .Output("bf_samples: T")
    .Doc(R"doc(
Beamforming
)doc");


template <typename T>
class BeamformingOp : public OpKernel {
 public:
  explicit BeamformingOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {


    //Read inputs
    const Tensor& focus_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(focus_tensor.shape()),
                errors::InvalidArgument("focus expects a 1-D vector."));
    auto focus = focus_tensor.flat<T>();

    const Tensor& center_focus_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(center_focus_tensor.shape()),
                errors::InvalidArgument("center_focus expects a 1D-vector."));
    auto center_focus = center_focus_tensor.flat<T>();

    const Tensor& speed_of_sound_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(speed_of_sound_tensor.shape()),
                errors::InvalidArgument("speed_of_sound expects a scalar."));
    auto speed_of_sound = speed_of_sound_tensor.flat<T>();

    const Tensor& t_start_tensor = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(t_start_tensor.shape()),
                errors::InvalidArgument("t_start expects a scalar."));
    auto t_start = t_start_tensor.flat<T>();

    const Tensor& t_start_data_tensor = context->input(4);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(t_start_data_tensor.shape()),
                errors::InvalidArgument("t_start_data expects a scalar."));
    auto t_start_data = t_start_data_tensor.flat<T>();


    const Tensor& f_sampling_tensor = context->input(5);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(f_sampling_tensor.shape()),
                errors::InvalidArgument("f_sampling expects a scalar."));
    auto f_sampling = f_sampling_tensor.flat<T>();

    const Tensor& f_number_tensor = context->input(6);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(f_number_tensor.shape()),
                errors::InvalidArgument("f_number expects a scalar."));
    auto f_number = f_number_tensor.flat<T>();


    const Tensor& element_pos_tensor = context->input(7);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(element_pos_tensor.shape()),
                errors::InvalidArgument("element_pos expects a matrix."));
    auto element_pos = element_pos_tensor.flat<T>();

    const Tensor& point_pos_tensor = context->input(8);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(point_pos_tensor.shape()),
                errors::InvalidArgument("point_pos expects a matrix."));
    auto point_pos = point_pos_tensor.flat<T>();

    const Tensor& samples_tensor = context->input(9);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(samples_tensor.shape()),
                errors::InvalidArgument("samples expects a matrix."));
    auto samples = samples_tensor.flat<T>();



    double wall_time = get_wall_time();



    //printf("point_pos: %d, %d\n", (int)point_pos_tensor.dim_size(0), (int)point_pos_tensor.dim_size(1));

	  

    // Output a float32 tensor.
    Tensor* bf_samples_tensor = NULL;
    int output_dimensions = 1;
    if(samples_tensor.dims() > 2)
    {
      output_dimensions = samples_tensor.dim_size(0);
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({samples_tensor.dim_size(0), point_pos_tensor.dim_size(0)}), &bf_samples_tensor));
    }
    else
    {
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({point_pos_tensor.dim_size(0)}), &bf_samples_tensor));
    }
    auto bf_samples = bf_samples_tensor->template flat<T>();






    // Distance center focus to center
    
    T tof_cf = 0;
    for (int i = 0; i < focus.size(); i++)
    {
      tof_cf += (focus(i) - center_focus(i)) * (focus(i) - center_focus(i));
    }
    tof_cf = sqrtf(tof_cf);
    printf("tof_cf: %f\n", tof_cf);



    //Loop through all points and get distance to virtual source
    const int points = point_pos_tensor.dim_size(0);
    const int elements = element_pos_tensor.dim_size(0);
    const int Dims = point_pos_tensor.dim_size(1);
    auto inv_sof = 1.0/speed_of_sound(0);

    const int sample_points = samples_tensor.dim_size(samples_tensor.dims()-2);
    const int sample_elements = samples_tensor.dim_size(samples_tensor.dims()-1);


    printf("points: %d, elements: %d\n", points, elements);

    auto max_index = (int)samples_tensor.dim_size(samples_tensor.dims()-2) - 2;


    #pragma omp parallel for
    for (int i = 0; i < points; i++) 
    {


      //Calc dist from point to VS(focus)
      T tof_tx = 0;
      for (int k = 0; k < Dims; k++)
      {
        tof_tx += (focus(k) - point_pos(i*Dims+k)) * (focus(k) - point_pos(i*Dims+k));
      }
      tof_tx = sqrtf(tof_tx);


      //FIND APOD VALUES
      T *ref_xyz = new T[Dims];
      T ref_dist = 1000000;
      T *tof_tmp = new T[elements];
      
      //      #pragma omp simd safelen(8)
      for(int j = 0; j < elements; j++)
      {
        //Calc dist from point to element
        T tof_rx = 0;
        for (int k = 0; k < Dims; k++)
        {
          tof_rx += (element_pos(j*Dims+k) - point_pos(i*Dims+k)) * (element_pos(j*Dims+k) - point_pos(i*Dims+k));
        }
        tof_rx = sqrtf(tof_rx);
	tof_tmp[j] = (tof_rx+tof_tx+tof_cf)*inv_sof - t_start(0);

	
        //Save closest distance to element and vector from element to bf_point
        if(tof_rx < ref_dist)
        { 
          ref_dist = tof_rx;
	  for (int k = 0; k < Dims; k++)
	  {
            ref_xyz[k] = element_pos(j*Dims+k) - point_pos(i*Dims+k);
	  }
        }
	
      }

      T* norm_ref = new T[Dims];
      for (int k = 0; k < Dims; k++)
      {
        norm_ref[k] = point_pos(i*Dims+k) + ref_xyz[k] / ref_dist;
      }

      //Calc distance to line for all elements

      ref_dist = ref_dist/f_number(0) * 0.5;
      //      if( i == 0)
      //	printf("i: %d, ref_dist: %f\n", i, ref_dist);


      //END FIND APOD VALUES

      for (int j = 0; j < elements; j++) 
      {
	
	T tof_idx_weight = (tof_tmp[j]-t_start_data(0))*f_sampling(0);
	int tof_idx = round(tof_idx_weight);
	tof_idx_weight = tof_idx_weight - tof_idx;
	tof_idx = std::max(0, std::min(tof_idx, max_index));
	int tof_idx_plus = tof_idx + 1;

        //if(i == 0 && j == 0)
	//  printf("tof_tmp: %e, %d, %d, %e, %e\n", tof_tmp, tof_idx, tof_idx_plus, t_start_data(0), f_sampling(0));

	//BEGIN APOD WEIGHT
        T* v1 = new T[Dims];
        T* v2 = new T[Dims];
        for (int k = 0; k < Dims; k++)
        {
          v1[k] = element_pos(j*Dims+k) - norm_ref[k];
          v2[k] = element_pos(j*Dims+k) - point_pos(i*Dims+k);
        }
        T* cross = new T[Dims];
	for( int k = 0; k < Dims; k++)
        {
          int firstIndex = (k+1)%Dims;
          int secondIndex = (k+2)%Dims;
	  cross[k] = v1[firstIndex] * v2[secondIndex] - v1[secondIndex] * v2[firstIndex];
        }
        T dist_apod = 0;
        for( int k = 0; k < Dims; k++)
        {
          dist_apod += cross[k]*cross[k];
        }
        dist_apod = sqrt(dist_apod);
        T dist_apod_norm = std::min((T)dist_apod/ref_dist, (T)1.0);
        T apod_weights=0.5*(1.0+cos(M_PI*dist_apod_norm));
	//END APOD WEIGHT

	
	for(int k = 0; k < output_dimensions; k++)
	{
	  int k_index = k*sample_points*sample_elements;
	  T sample_idx = samples(k_index + tof_idx*elements+j);
	  T sample_idx_plus = samples(k_index + tof_idx_plus*elements+j);
	  T delayed_sample = sample_idx + (sample_idx_plus - sample_idx) * tof_idx_weight;
	  if(j == 0) bf_samples(i + k*points) = 0;
	  bf_samples(i + k*points) += delayed_sample*apod_weights;
	  //if(i == 0 && j == 0 && k == 0) printf("k %d, samp: %e, w: %e, s: %e, sp: %e, t: %d, tp: %d, index: %d, apod: %e\n", k, delayed_sample, tof_idx_weight,  sample_idx, sample_idx_plus, tof_idx*elements+j, tof_idx_plus*elements+j, k_index + tof_idx*elements+j, apod_weights);

	}
	
        delete[] v1;
        delete[] v2;
        delete[] cross;

      }
    
      delete[] norm_ref;

      delete[] ref_xyz;
      delete[] tof_tmp;
    }
    //for(int i = 0; i < 100; i++)
    //  {
    //	printf("%e\n", tof_pos(i));
    //  }


    std::cout << "Beamforming (CPU): " << (get_wall_time() - wall_time)*1000.0 << " ms" << std::endl;

  }
};

#define REGISTER_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(						 \
      Name("Beamforming").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BeamformingOp<type>)

REGISTER_KERNEL(double);
REGISTER_KERNEL(float);

#undef REGISTER_KERNEL


REGISTER_OP("BeamformingAvx")
    .Attr("T: {double, float}")
    .Input("focus: T")
    .Input("center_focus: T")
    .Input("speed_of_sound: T")
    .Input("t_start: T")
    .Input("t_start_data: T")
    .Input("f_sampling: T")
    .Input("f_number: T")
    .Input("element_pos: T")
    .Input("point_pos: T")
    .Input("samples: T")
    .Output("bf_samples: T")
    .Doc(R"doc(
Beamforming AVX
)doc");

template <typename T>
class BeamformingAVXOp : public OpKernel {
 public:
  explicit BeamformingAVXOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {}
};



template <>
class BeamformingAVXOp<float> : public OpKernel {
 public:
  explicit BeamformingAVXOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {


    //Read inputs
    const Tensor& focus_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(focus_tensor.shape()),
                errors::InvalidArgument("focus expects a 1-D vector."));
    auto focus = focus_tensor.flat<float>();

    const Tensor& center_focus_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(center_focus_tensor.shape()),
                errors::InvalidArgument("center_focus expects a 1D-vector."));
    auto center_focus = center_focus_tensor.flat<float>();

    const Tensor& speed_of_sound_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(speed_of_sound_tensor.shape()),
                errors::InvalidArgument("speed_of_sound expects a scalar."));
    auto speed_of_sound = speed_of_sound_tensor.flat<float>();

    const Tensor& t_start_tensor = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(t_start_tensor.shape()),
                errors::InvalidArgument("t_start expects a scalar."));
    auto t_start = t_start_tensor.flat<float>();

    const Tensor& t_start_data_tensor = context->input(4);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(t_start_data_tensor.shape()),
                errors::InvalidArgument("t_start_data expects a scalar."));
    auto t_start_data = t_start_data_tensor.flat<float>();


    const Tensor& f_sampling_tensor = context->input(5);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(f_sampling_tensor.shape()),
                errors::InvalidArgument("f_sampling expects a scalar."));
    auto f_sampling = f_sampling_tensor.flat<float>();

    const Tensor& f_number_tensor = context->input(6);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(f_number_tensor.shape()),
                errors::InvalidArgument("f_number expects a scalar."));
    auto f_number = f_number_tensor.flat<float>();


    const Tensor& element_pos_tensor = context->input(7);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(element_pos_tensor.shape()),
                errors::InvalidArgument("element_pos expects a matrix."));
    auto element_pos = element_pos_tensor.flat<float>();

    const Tensor& point_pos_tensor = context->input(8);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(point_pos_tensor.shape()),
                errors::InvalidArgument("point_pos expects a matrix."));
    auto point_pos = point_pos_tensor.flat<float>();

    const Tensor& samples_tensor = context->input(9);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(samples_tensor.shape()),
                errors::InvalidArgument("samples expects a matrix."));
    auto samples = samples_tensor.flat<float>();



    double wall_time = get_wall_time();



    //printf("point_pos: %d, %d\n", (int)point_pos_tensor.dim_size(0), (int)point_pos_tensor.dim_size(1));

	  

    // Output a float32 tensor.
    Tensor* bf_samples_tensor = NULL;
    int output_dimensions = 1;
    if(samples_tensor.dims() > 2)
    {
      output_dimensions = samples_tensor.dim_size(0);
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({samples_tensor.dim_size(0), point_pos_tensor.dim_size(0)}), &bf_samples_tensor));
    }
    else
    {
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({point_pos_tensor.dim_size(0)}), &bf_samples_tensor));
    }
    auto bf_samples = bf_samples_tensor->template flat<float>();






    // Distance center focus to center
    
    float tof_cf = 0;
    for (int i = 0; i < focus.size(); i++)
    {
      tof_cf += (focus(i) - center_focus(i)) * (focus(i) - center_focus(i));
    }
    tof_cf = sqrtf(tof_cf);
    printf("tof_cf: %f\n", tof_cf);



    //Loop through all points and get distance to virtual source
    const int points = point_pos_tensor.dim_size(0);
    const int elements = element_pos_tensor.dim_size(0);
    const int Dims = point_pos_tensor.dim_size(1);
    auto inv_sof = 1.0/speed_of_sound(0);

    const int sample_points = samples_tensor.dim_size(samples_tensor.dims()-2);
    const int sample_elements = samples_tensor.dim_size(samples_tensor.dims()-1);


    printf("points: %d, elements: %d\n", points, elements);

    auto max_index = (int)samples_tensor.dim_size(samples_tensor.dims()-2) - 2;


    float *elements_reorder = new float[elements*Dims];
    for(int i = 0; i < Dims; i++) {
      for(int j = 0; j < elements; j++) {
	elements_reorder[j + i*elements] = element_pos(j*Dims + i);
      }
    }

    float *samples_reorder = new float[output_dimensions*sample_points*sample_elements];
    for(int i = 0; i < sample_points; i++) {
      for(int j = 0; j < sample_elements; j++) {
	for(int k = 0; k < output_dimensions; k++) {
	  samples_reorder[k + j*output_dimensions + i*sample_elements*output_dimensions] = samples(k*sample_elements*sample_points + i*sample_elements + j);
	}
      }
    }


    float *output_samples = new float[output_dimensions*points];
    memset(output_samples, 0, output_dimensions*points);

      __m256 plus_m = _mm256_set1_ps(1);
      __m256 pi_m = _mm256_set1_ps(M_PI);



    #pragma omp parallel for
    for (int i = 0; i < points; i++) 
    {

        float *apod_weights = new float[elements];
	

      //Calc dist from point to VS(focus)
      float tof_tx = 0;
      for (int k = 0; k < Dims; k++)
      {
        tof_tx += (focus(k) - point_pos(i*Dims+k)) * (focus(k) - point_pos(i*Dims+k));
      }
      tof_tx = sqrtf(tof_tx);


      //FIND APOD VALUES
      float *ref_xyz = new float[Dims];
      float ref_dist = 1000000;
      float *tof_tmp = new float[elements];

      __m256 ref_dist_m = _mm256_set1_ps(1000000);
      __m256 inv_sof_m = _mm256_set1_ps(inv_sof);
      __m256 t_start_m = _mm256_set1_ps(t_start(0));
      __m256 tof_tx_m = _mm256_set1_ps(tof_tx);
      __m256 tof_cf_m = _mm256_set1_ps(tof_cf);
      
      for(int j = 0; j < elements; j+=8)
      {
        //Calc dist from point to element
        __m256 tof_rx_m = _mm256_setzero_ps();
        for (int k = 0; k < Dims; k++)
	{
          __m256 point_pos_m = _mm256_set1_ps(point_pos(i*Dims+k));
	  __m256 element_pos_t_m = _mm256_loadu_ps(&elements_reorder[j + elements*k]);
	  tof_rx_m = _mm256_add_ps(tof_rx_m, _mm256_mul_ps(_mm256_sub_ps(element_pos_t_m, point_pos_m), _mm256_sub_ps(element_pos_t_m, point_pos_m)));
        }
        tof_rx_m = _mm256_sqrt_ps(tof_rx_m);
	float tof_rx_tmp[8];
	_mm256_storeu_ps(&tof_rx_tmp[0], tof_rx_m);

	__m256 tof_tmp_m = _mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(tof_rx_m, tof_tx_m), tof_cf_m), inv_sof_m), t_start_m);
	_mm256_storeu_ps(&tof_tmp[j], tof_tmp_m);

	for(int x = 0; x < 8; x++)
        {
          if(tof_rx_tmp[x] < ref_dist)
          {   
            ref_dist = tof_rx_tmp[x];
	    for (int k = 0; k < Dims; k++)
	    {
              ref_xyz[k] = elements_reorder[(j+x) + elements*k] - point_pos(i*Dims+k);
	    }
          }
        }

      }
     

      float* norm_ref = new float[Dims];
      for (int k = 0; k < Dims; k++)
      {
        norm_ref[k] = point_pos(i*Dims+k) + ref_xyz[k] / ref_dist;
      }

      //Calc distance to line for all elements

      ref_dist = ref_dist/f_number(0) * 0.5;
      //      if( i == 0)
      //	printf("i: %d, ref_dist: %f\n", i, ref_dist);


      //END FIND APOD VALUES

	ref_dist_m = _mm256_set1_ps(ref_dist);
	for (int j = 0; j < elements; j+=8) 
        {
          //T dist_apod = 0;
          __m256 dist_apod_m = _mm256_setzero_ps();
	  for( int k = 0; k < Dims; k++)
	  {
            int fI = (k+1)%Dims;
      	    int sI = (k+2)%Dims;

	    __m256 norm_ref_fI_m = _mm256_set1_ps(norm_ref[fI]);
	    __m256 norm_ref_sI_m = _mm256_set1_ps(norm_ref[sI]);

	    __m256 e_fI_m = _mm256_loadu_ps(&elements_reorder[j + (fI*elements)]);
	    __m256 e_sI_m = _mm256_loadu_ps(&elements_reorder[j + (sI*elements)]);

	    __m256 p_fI_m = _mm256_set1_ps(point_pos(i*Dims+fI));
	    __m256 p_sI_m = _mm256_set1_ps(point_pos(i*Dims+sI));

	    __m256 v1_m = _mm256_sub_ps(e_fI_m, norm_ref_fI_m);
	    __m256 v2_m = _mm256_sub_ps(e_sI_m, p_sI_m);
	    __m256 v3_m = _mm256_sub_ps(e_sI_m, norm_ref_sI_m);
	    __m256 v4_m = _mm256_sub_ps(e_fI_m, p_fI_m);

	    __m256 cross_m = _mm256_sub_ps(_mm256_mul_ps(v1_m, v2_m), _mm256_mul_ps(v3_m, v4_m));
	    cross_m = _mm256_mul_ps(cross_m, cross_m);
	    dist_apod_m = _mm256_add_ps(dist_apod_m, cross_m);

          }
          dist_apod_m = _mm256_sqrt_ps(dist_apod_m);

	  __m256 dist_apod_norm_m = _mm256_min_ps(_mm256_div_ps(dist_apod_m, ref_dist_m), plus_m);
	  __m256 apod_weights_m = _mm256_mul_ps(dist_apod_norm_m, pi_m);

	  float weights[8];
	  _mm256_storeu_ps(&weights[0], apod_weights_m);
	
	  //apod_weights[j] = apod_weight;
	  apod_weights[j] = 0.5*(1.0+cos(weights[0]));
	  apod_weights[j+1] = 0.5*(1.0+cos(weights[1]));
	  apod_weights[j+2] = 0.5*(1.0+cos(weights[2]));
	  apod_weights[j+3] = 0.5*(1.0+cos(weights[3]));
	  apod_weights[j+4] = 0.5*(1.0+cos(weights[4]));
	  apod_weights[j+5] = 0.5*(1.0+cos(weights[5]));
	  apod_weights[j+6] = 0.5*(1.0+cos(weights[6]));
	  apod_weights[j+7] = 0.5*(1.0+cos(weights[7]));

        }
	//END APOD WEIGHT

	__m256 t_start_data_m = _mm256_set1_ps(t_start_data(0));
	__m256 f_sampling_m = _mm256_set1_ps(f_sampling(0));
	__m256 max_index_m = _mm256_set1_ps(max_index);
	__m256 min_index_m = _mm256_setzero_ps();

	for (int j = 0; j < elements; j+=8) 
	{
          __m256 tof_tmp_m = _mm256_loadu_ps(&tof_tmp[j]);
	  __m256 tof_idx_weight_m = _mm256_mul_ps(_mm256_sub_ps(tof_tmp_m, t_start_data_m), f_sampling_m);
	  __m256 tof_idx_m = _mm256_round_ps(tof_idx_weight_m, 0x08);
	  tof_idx_weight_m = _mm256_sub_ps(tof_idx_weight_m, tof_idx_m);
	  tof_idx_m = _mm256_max_ps(min_index_m, _mm256_min_ps(tof_idx_m, max_index_m));
	  float tof_idx_weight_a[8];
	  float tof_idx_a[8];
	  _mm256_storeu_ps(&tof_idx_a[0], tof_idx_m);
	  _mm256_storeu_ps(&tof_idx_weight_a[0], tof_idx_weight_m);

	  for(int x = 0; x < 8; x++) 
	  {
	    __m256 tof_idx_weight_a_m = _mm256_set1_ps(tof_idx_weight_a[x]);
	  
	    for(int k = 0; k < output_dimensions; k+=8) 
	    {
              __m256 sample_idx_t_m = _mm256_loadu_ps(&samples_reorder[((int)tof_idx_a[x])*elements*output_dimensions + (j+x)*output_dimensions + k]);
	      __m256 sample_idx_plus_t_m = _mm256_loadu_ps(&samples_reorder[(((int)tof_idx_a[x])+1)*elements*output_dimensions + (j+x)*output_dimensions + k]);
	    
	      __m256 apod_weights_t_m = _mm256_set1_ps(apod_weights[j+x]);
	    
	      __m256 delayed_sample_m = _mm256_mul_ps(_mm256_add_ps(sample_idx_t_m, _mm256_mul_ps(_mm256_sub_ps(sample_idx_plus_t_m, sample_idx_t_m), tof_idx_weight_a_m)), apod_weights_t_m);
	    
	      __m256 add_samples_m = _mm256_add_ps(delayed_sample_m, _mm256_loadu_ps(&output_samples[i*output_dimensions + k]));
	      _mm256_storeu_ps(&output_samples[i*output_dimensions +k], add_samples_m);
	    }
	  }
        }
      
	delete[] norm_ref;
	delete[] apod_weights;
	delete[] ref_xyz;
	delete[] tof_tmp;
      }



    delete[] samples_reorder;
    delete[] elements_reorder;

    //Reorder output
    for(int i = 0; i < output_dimensions; i++) {
      for(int j = 0; j < points; j++) {
	bf_samples(j + i*points) = output_samples[i + j*output_dimensions];
	//if(i == 0 && j < 16) printf("%d, %e\n", j, bf_samples(j+i*points));
      }
    }
    delete[] output_samples;
     

    std::cout << "Beamforming (AVX - float): " << (get_wall_time() - wall_time)*1000.0 << " ms" << std::endl;

  }
};

#define REGISTER_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(						 \
      Name("BeamformingAvx").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BeamformingAVXOp<type>)

REGISTER_KERNEL(float);

#undef REGISTER_KERNEL




template <>
class BeamformingAVXOp<double> : public OpKernel {
 public:
  explicit BeamformingAVXOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {


    //Read inputs
    const Tensor& focus_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(focus_tensor.shape()),
                errors::InvalidArgument("focus expects a 1-D vector."));
    auto focus = focus_tensor.flat<double>();

    const Tensor& center_focus_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(center_focus_tensor.shape()),
                errors::InvalidArgument("center_focus expects a 1D-vector."));
    auto center_focus = center_focus_tensor.flat<double>();

    const Tensor& speed_of_sound_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(speed_of_sound_tensor.shape()),
                errors::InvalidArgument("speed_of_sound expects a scalar."));
    auto speed_of_sound = speed_of_sound_tensor.flat<double>();

    const Tensor& t_start_tensor = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(t_start_tensor.shape()),
                errors::InvalidArgument("t_start expects a scalar."));
    auto t_start = t_start_tensor.flat<double>();

    const Tensor& t_start_data_tensor = context->input(4);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(t_start_data_tensor.shape()),
                errors::InvalidArgument("t_start_data expects a scalar."));
    auto t_start_data = t_start_data_tensor.flat<double>();


    const Tensor& f_sampling_tensor = context->input(5);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(f_sampling_tensor.shape()),
                errors::InvalidArgument("f_sampling expects a scalar."));
    auto f_sampling = f_sampling_tensor.flat<double>();

    const Tensor& f_number_tensor = context->input(6);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(f_number_tensor.shape()),
                errors::InvalidArgument("f_number expects a scalar."));
    auto f_number = f_number_tensor.flat<double>();


    const Tensor& element_pos_tensor = context->input(7);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(element_pos_tensor.shape()),
                errors::InvalidArgument("element_pos expects a matrix."));
    auto element_pos = element_pos_tensor.flat<double>();

    const Tensor& point_pos_tensor = context->input(8);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(point_pos_tensor.shape()),
                errors::InvalidArgument("point_pos expects a matrix."));
    auto point_pos = point_pos_tensor.flat<double>();

    const Tensor& samples_tensor = context->input(9);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(samples_tensor.shape()),
                errors::InvalidArgument("samples expects a matrix."));
    auto samples = samples_tensor.flat<double>();



    double wall_time = get_wall_time();



    //printf("point_pos: %d, %d\n", (int)point_pos_tensor.dim_size(0), (int)point_pos_tensor.dim_size(1));

	  

    // Output a float32 tensor.
    Tensor* bf_samples_tensor = NULL;
    int output_dimensions = 1;
    if(samples_tensor.dims() > 2)
    {
      output_dimensions = samples_tensor.dim_size(0);
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({samples_tensor.dim_size(0), point_pos_tensor.dim_size(0)}), &bf_samples_tensor));
    }
    else
    {
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({point_pos_tensor.dim_size(0)}), &bf_samples_tensor));
    }
    auto bf_samples = bf_samples_tensor->template flat<double>();


    

    // Distance center focus to center
    
    double tof_cf = 0;
    for (int i = 0; i < focus.size(); i++)
    {
      tof_cf += (focus(i) - center_focus(i)) * (focus(i) - center_focus(i));
    }
    tof_cf = sqrtf(tof_cf);
    printf("tof_cf: %f\n", tof_cf);



    //Loop through all points and get distance to virtual source
    const int points = point_pos_tensor.dim_size(0);
    const int elements = element_pos_tensor.dim_size(0);
    const int Dims = point_pos_tensor.dim_size(1);
    auto inv_sof = 1.0/speed_of_sound(0);

    const int sample_points = samples_tensor.dim_size(samples_tensor.dims()-2);
    const int sample_elements = samples_tensor.dim_size(samples_tensor.dims()-1);


    printf("points: %d, elements: %d\n", points, elements);

    auto max_index = (int)samples_tensor.dim_size(samples_tensor.dims()-2) - 2;

    double *samples_reorder = new double[output_dimensions*sample_points*sample_elements];
    for(int i = 0; i < sample_points; i++) {
      for(int j = 0; j < sample_elements; j++) {
	for(int k = 0; k < output_dimensions; k++) {
	  samples_reorder[k + j*output_dimensions + i*sample_elements*output_dimensions] = samples(k*sample_elements*sample_points + i*sample_elements + j);
	}
      }
    }


    double *elements_reorder = new double[elements*Dims];
    for(int i = 0; i < Dims; i++) {
      for(int j = 0; j < elements; j++) {
	elements_reorder[j + i*elements] = element_pos(j*Dims + i);
      }
    }

 

    double *output_samples = new double[output_dimensions*points];
    memset(output_samples, 0, output_dimensions*points);
 
      __m256d plus_m = _mm256_set1_pd(1);
      __m256d pi_m = _mm256_set1_pd(M_PI);

    


       #pragma omp parallel for
      for (int i = 0; i < points; i++) 
      {
	//Calc dist from point to VS(focus)
	double tof_tx = 0;
	for (int k = 0; k < Dims; k++)
	{
	  tof_tx += (focus(k) - point_pos(i*Dims+k)) * (focus(k) - point_pos(i*Dims+k));
	}
	tof_tx = sqrtf(tof_tx);


	//FIND APOD VALUES
	double *ref_xyz = new double[Dims];

	double ref_dist = 1000000;
	double *tof_tmp = new double[elements];
	double *apod_weights = new double[elements];
      
	__m256d ref_dist_m = _mm256_set1_pd(1000000);
	__m256d inv_sof_m = _mm256_set1_pd(inv_sof);
	__m256d t_start_m = _mm256_set1_pd(t_start(0));
	__m256d tof_tx_m = _mm256_set1_pd(tof_tx);
	__m256d tof_cf_m = _mm256_set1_pd(tof_cf);
	
	for(int j = 0; j < elements; j+=4)
	{
	  //Calc dist from point to element
	  __m256d tof_rx_m = _mm256_setzero_pd();
	  for (int k = 0; k < Dims; k++)
	  {
	    __m256d point_pos_m = _mm256_set1_pd(point_pos(i*Dims+k));
	    __m256d element_pos_t_m = _mm256_loadu_pd(&elements_reorder[j + elements*k]);
	    tof_rx_m = _mm256_add_pd(tof_rx_m, _mm256_mul_pd(_mm256_sub_pd(element_pos_t_m, point_pos_m), _mm256_sub_pd(element_pos_t_m, point_pos_m)));
	  }
	  tof_rx_m = _mm256_sqrt_pd(tof_rx_m);
	  double tof_rx_tmp[4];
	  _mm256_storeu_pd(&tof_rx_tmp[0], tof_rx_m);

	  __m256d tof_tmp_m = _mm256_sub_pd(_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(tof_rx_m, tof_tx_m), tof_cf_m), inv_sof_m), t_start_m);
	  _mm256_storeu_pd(&tof_tmp[j], tof_tmp_m);

	  //Save closest distance to element and vector from element to bf_point
	
	  //Permute input = [A B C D] to vec1 = [C D A B]
	  __m256d vec1 = _mm256_permute2f128_pd(tof_rx_m, tof_rx_m, 1);
	  //Find max val of input and vec1 = [min(A,C), min(B,D), min(C,A), min(D,B)]
	  __m256d min1 = _mm256_min_pd(tof_rx_m, vec1);
	  //Permute max1 to [min(B,D), min(A,C), min(D, B), min(C,A)]
	  __m256d vec2 = _mm256_permute_pd(min1, 5);
	  //Find max of min1 and vec2 = [ min(min(A,C), min(B, D)), etc... ]
	  __m256d min2 = _mm256_min_pd(min1, vec2);
	  
	  //Compare ref_dist to tof_rx
	  __m256d cmp_val = _mm256_cmp_pd(ref_dist_m, tof_rx_m, _CMP_LT_OQ);
	  //Save new ref_dist if smaller value found in tof_rx
	  ref_dist_m = _mm256_blendv_pd(tof_rx_m, ref_dist_m, cmp_val);
	  
	  //"Splat" maxval to all indexes
	  __m256d splatval = _mm256_permute_pd(_mm256_permute2f128_pd(min2,min2, 49), 15);
	  //find new minimum index by comparing if the min value is equal to index in ref_dist
	  int index = _mm256_movemask_pd(_mm256_cmp_pd(splatval, ref_dist_m, _CMP_EQ_OQ));
	  int new_index = (ffs(index)-1);
	  if(new_index >= 0 && tof_rx_tmp[new_index] < ref_dist)
	  {
	    ref_dist = tof_rx_tmp[new_index];
	    for (int k = 0; k < Dims; k++)
	    {
	      ref_xyz[k] = element_pos((j+new_index)*Dims+k) - point_pos(i*Dims+k);
	    }
	  }
	 
	  if(i == 0) 
	  {
	    printf("%d, %d, %e, %e\n", j, new_index, tof_rx_tmp[new_index], ref_dist); 

          }

 
	}


	double* norm_ref = new double[Dims];
	for (int k = 0; k < Dims; k++)
	{
	  norm_ref[k] = point_pos(i*Dims+k) + ref_xyz[k] / ref_dist;
	}

	//Calc distance to line for all elements

	ref_dist = ref_dist/f_number(0) * 0.5;
	
	//END FIND APOD VALUES

	ref_dist_m = _mm256_set1_pd(ref_dist);
	for (int j = 0; j < elements; j+=4) 
	{
	  //T dist_apod = 0;
	  __m256d dist_apod_m = _mm256_setzero_pd();
	  for( int k = 0; k < Dims; k++)
	  {
	    int fI = (k+1)%Dims;
	    int sI = (k+2)%Dims;
	    
	    __m256d norm_ref_fI_m = _mm256_set1_pd(norm_ref[fI]);
	    __m256d norm_ref_sI_m = _mm256_set1_pd(norm_ref[sI]);
	    
	    __m256d e_fI_m = _mm256_loadu_pd(&elements_reorder[j + (fI*elements)]);
	    __m256d e_sI_m = _mm256_loadu_pd(&elements_reorder[j + (sI*elements)]);
	    
	    __m256d p_fI_m = _mm256_set1_pd(point_pos(i*Dims+fI));
	    __m256d p_sI_m = _mm256_set1_pd(point_pos(i*Dims+sI));
	    
	    __m256d v1_m = _mm256_sub_pd(e_fI_m, norm_ref_fI_m);
	    __m256d v2_m = _mm256_sub_pd(e_sI_m, p_sI_m);
	    __m256d v3_m = _mm256_sub_pd(e_sI_m, norm_ref_sI_m);
	    __m256d v4_m = _mm256_sub_pd(e_fI_m, p_fI_m);
	    
	    __m256d cross_m = _mm256_sub_pd(_mm256_mul_pd(v1_m, v2_m), _mm256_mul_pd(v3_m, v4_m));
	    cross_m = _mm256_mul_pd(cross_m, cross_m);
	    dist_apod_m = _mm256_add_pd(dist_apod_m, cross_m);
	    
	  }
	  dist_apod_m = _mm256_sqrt_pd(dist_apod_m);
	  
	  __m256d dist_apod_norm_m = _mm256_min_pd(_mm256_div_pd(dist_apod_m, ref_dist_m), plus_m);
	  __m256d apod_weights_m = _mm256_mul_pd(dist_apod_norm_m, pi_m);
	  
	  double weights[4];
	  _mm256_storeu_pd(&weights[0], apod_weights_m);
	  
	  //apod_weights[j] = apod_weight;
	  apod_weights[j] = 0.5*(1.0+cos(weights[0]));
	  apod_weights[j+1] = 0.5*(1.0+cos(weights[1]));
	  apod_weights[j+2] = 0.5*(1.0+cos(weights[2]));
	  apod_weights[j+3] = 0.5*(1.0+cos(weights[3]));

	}
	//END APOD WEIGHT

	__m256d t_start_data_m = _mm256_set1_pd(t_start_data(0));
	__m256d f_sampling_m = _mm256_set1_pd(f_sampling(0));
	__m256d max_index_m = _mm256_set1_pd(max_index);
	__m256d min_index_m = _mm256_setzero_pd();
	
	for (int j = 0; j < elements; j+=4) 
	{
	  __m256d tof_tmp_m = _mm256_loadu_pd(&tof_tmp[j]);
	  __m256d tof_idx_weight_m = _mm256_mul_pd(_mm256_sub_pd(tof_tmp_m, t_start_data_m), f_sampling_m);
	  __m256d tof_idx_m = _mm256_round_pd(tof_idx_weight_m, 0x08);
	  tof_idx_weight_m = _mm256_sub_pd(tof_idx_weight_m, tof_idx_m);
	  tof_idx_m = _mm256_max_pd(min_index_m, _mm256_min_pd(tof_idx_m, max_index_m));
	  double tof_idx_weight_a[4];
	  double tof_idx_a[4];
	  _mm256_storeu_pd(&tof_idx_a[0], tof_idx_m);
	  _mm256_storeu_pd(&tof_idx_weight_a[0], tof_idx_weight_m);
	  
	  for(int x = 0; x < 4; x++) 
	  {
	    __m256d tof_idx_weight_a_m = _mm256_set1_pd(tof_idx_weight_a[x]);
	    
	    for(int k = 0; k < output_dimensions; k+=4) 
	    {
	      __m256d sample_idx_t_m = _mm256_loadu_pd(&samples_reorder[((int)tof_idx_a[x])*elements*output_dimensions + (j+x)*output_dimensions + k]);
	      __m256d sample_idx_plus_t_m = _mm256_loadu_pd(&samples_reorder[(((int)tof_idx_a[x])+1)*elements*output_dimensions + (j+x)*output_dimensions + k]);
	      
	      __m256d apod_weights_t_m = _mm256_set1_pd(apod_weights[j+x]);
	      
	      __m256d delayed_sample_m = _mm256_mul_pd(_mm256_add_pd(sample_idx_t_m, _mm256_mul_pd(_mm256_sub_pd(sample_idx_plus_t_m, sample_idx_t_m), tof_idx_weight_a_m)), apod_weights_t_m);
	      
	      __m256d add_samples_m = _mm256_add_pd(delayed_sample_m, _mm256_loadu_pd(&output_samples[i*output_dimensions + k]));
	      _mm256_storeu_pd(&output_samples[i*output_dimensions +k], add_samples_m);
	    }
	  }
	}
	
	delete[] norm_ref;
	delete[] apod_weights;
	delete[] ref_xyz;
	delete[] tof_tmp;
      }
    
    delete[] samples_reorder;
    delete[] elements_reorder;

    //Reorder output
    for(int i = 0; i < output_dimensions; i++) {
      for(int j = 0; j < points; j++) {
	bf_samples(j + i*points) = output_samples[i + j*output_dimensions];
	//if(i == 0 && j < 16) printf("%d, %e\n", j, bf_samples(j+i*points));
      }
    }
    delete[] output_samples;
    
    std::cout << "Beamforming (AVX): " << (get_wall_time() - wall_time)*1000.0 << " ms" << std::endl;

  }
};

#define REGISTER_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(						 \
      Name("BeamformingAvx").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BeamformingAVXOp<type>)

REGISTER_KERNEL(double);

#undef REGISTER_KERNEL




#if GOOGLE_CUDA

template <typename T>
bool LaunchBeamformingKernel(int elements, int points, int batch_size, int sample_size, int max_index, 
			     const T *focus, const T *center_focus, 
			     const T* speed_of_sound, const T* t_start, const T* t_start_data, 
			     const T* f_sampling, const T* f_number, const T* element_pos, 
			     const T* point_pos, const T* samples, T* bf_samples);


template <typename T>
class BeamformingGPUOp : public OpKernel {
 public:
  explicit BeamformingGPUOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {


    //Read inputs
    const Tensor& focus_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(focus_tensor.shape()),
                errors::InvalidArgument("focus expects a 1-D vector."));
    auto focus = focus_tensor.flat<T>();

    const Tensor& center_focus_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(center_focus_tensor.shape()),
                errors::InvalidArgument("center_focus expects a 1D-vector."));
    auto center_focus = center_focus_tensor.flat<T>();

    const Tensor& speed_of_sound_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(speed_of_sound_tensor.shape()),
                errors::InvalidArgument("speed_of_sound expects a scalar."));
    auto speed_of_sound = speed_of_sound_tensor.flat<T>();

    const Tensor& t_start_tensor = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(t_start_tensor.shape()),
                errors::InvalidArgument("t_start expects a scalar."));
    auto t_start = t_start_tensor.flat<T>();

    const Tensor& t_start_data_tensor = context->input(4);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(t_start_data_tensor.shape()),
                errors::InvalidArgument("t_start_data expects a scalar."));
    auto t_start_data = t_start_data_tensor.flat<T>();


    const Tensor& f_sampling_tensor = context->input(5);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(f_sampling_tensor.shape()),
                errors::InvalidArgument("f_sampling expects a scalar."));
    auto f_sampling = f_sampling_tensor.flat<T>();

    const Tensor& f_number_tensor = context->input(6);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(f_number_tensor.shape()),
                errors::InvalidArgument("f_number expects a scalar."));
    auto f_number = f_number_tensor.flat<T>();


    const Tensor& element_pos_tensor = context->input(7);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(element_pos_tensor.shape()),
                errors::InvalidArgument("element_pos expects a matrix."));
    auto element_pos = element_pos_tensor.flat<T>();

    const Tensor& point_pos_tensor = context->input(8);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(point_pos_tensor.shape()),
                errors::InvalidArgument("point_pos expects a matrix."));
    auto point_pos = point_pos_tensor.flat<T>();

    const Tensor& samples_tensor = context->input(9);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(samples_tensor.shape()),
                errors::InvalidArgument("samples expects a matrix."));
    auto samples = samples_tensor.flat<T>();



    double wall_time = get_wall_time();



    //Loop through all points and get distance to virtual source
    const int points = point_pos_tensor.dim_size(0);
    const int elements = element_pos_tensor.dim_size(0);
    const int Dims = point_pos_tensor.dim_size(1);
    auto inv_sof = 1.0/speed_of_sound(0);

    printf("point_pos: %d, %d\n", (int)point_pos_tensor.dim_size(0), (int)point_pos_tensor.dim_size(1));
	  

    // Output a float32 tensor.
    Tensor* bf_samples_tensor = NULL;
    int output_dimensions = 1;
    if(samples_tensor.dims() > 2)
    {
      output_dimensions = samples_tensor.dim_size(0);
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({samples_tensor.dim_size(0), point_pos_tensor.dim_size(0)}), &bf_samples_tensor));
    }
    else
    {
      OP_REQUIRES_OK(context,
		     context->allocate_output(0, TensorShape({point_pos_tensor.dim_size(0)}), &bf_samples_tensor));
    }
    auto bf_samples = bf_samples_tensor->template flat<T>();

    const int  max_index = (int)samples_tensor.dim_size(samples_tensor.dims()-2) - 2;
    const int sample_size = samples_tensor.dim_size(samples_tensor.dims()-2) * samples_tensor.dim_size(samples_tensor.dims()-1);

    // Distance center focus to center
    


    OP_REQUIRES(context, LaunchBeamformingKernel(element_pos_tensor.dim_size(0), point_pos_tensor.dim_size(0), samples_tensor.dim_size(0), 
						 sample_size, max_index, focus.data(), center_focus.data(), 
			    speed_of_sound.data(), t_start.data(), t_start_data.data(), f_sampling.data(), 
			    f_number.data(), element_pos.data(), point_pos.data(), 
						 samples.data(), bf_samples.data()),
		errors::Internal("LaunchBeamformingKernel() failed."));


    std::cout << "Beamforming (GPU): " << (get_wall_time() - wall_time)*1000.0 << " ms" << std::endl;


  }
};

#define REGISTER_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(						 \
      Name("Beamforming").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BeamformingGPUOp<type>)

REGISTER_KERNEL(double);
REGISTER_KERNEL(float);

#undef REGISTER_KERNEL

#endif
