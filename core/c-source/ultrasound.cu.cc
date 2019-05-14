/*
 
 GPU beamforming kernel
 
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
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void AddOneKernel(const int* in, const int N, int* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    out[i] = in[i] + 1;
  }
}

void AddOneKernelLauncher(const int* in, const int N, int* out) {
  AddOneKernel<<<32, 256>>>(in, N, out);
}


template <typename T, int BLOCK_DIM_X>
__global__ void BeamformingKernel(int elements, int points, int batch_size, int sample_size, int max_index, 
				  const T *focus, const T* center_focus, 
				  const T* speed_of_sound, const T* t_start, const T* t_start_data, 
				  const T* f_sampling, const T* f_number, const T* element_pos, 
				  const T* point_pos, const T* samples, T* bf_samples) 
{
  const int i = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
  if (i < points) 
  {

    float tof_cf = 0.028800;
    /*
    for (int j = 0; j < 3; j++)
    {
      tof_cf += (focus[j] - center_focus[j]) * (focus[j] - center_focus[j]);
    }
    tof_cf = sqrtf(tof_cf);
    */

    T inv_sof = 0.0006493506493506494;//1.0/speed_of_sound[0];


      //Calc dist from point to VS(focus)
      //T tof_tx = 0;
      //int Dims = 3;
      //for (int k = 0; k < Dims; k++)
      //{
      //tof_tx += (focus[0] - point_pos[i*3]) * (focus[0] - point_pos[i*3]);
      // tof_tx += (focus[1] - point_pos[i*3+1]) * (focus[1] - point_pos[i*3+1]);
      //tof_tx += (focus[2] - point_pos[i*3+2]) * (focus[2] - point_pos[i*3+2]);
	//}
        //tof_tx = sqrtf(tof_tx);
	T tof_tx = norm3d(focus[0] - point_pos[i*3], focus[1] - point_pos[i*3+1], focus[2] - point_pos[i*3+2]);
       

      //FIND APOD VALUES
      T ref_xyz[3];// = new T[Dims];
      T ref_dist = 1000000;
      for(int j = 0; j < elements; j++)
      {
        //Calc dist from point to element
        //T tof_rx = 0;
        //for (int k = 0; k < Dims; k++)
        //{
        //  tof_rx += (element_pos[j*Dims+k] - point_pos[i*Dims+k]) * (element_pos[j*Dims+k] - point_pos[i*Dims+k]);
        //}
        //tof_rx = sqrtf(tof_rx);
	T tof_rx = norm3d(element_pos[j*3] - point_pos[i*3], element_pos[j*3+1] - point_pos[i*3+1], element_pos[j*3+2] - point_pos[i*3+2]);
        //Save closest distance to element and vector from element to bf_point
        if(tof_rx < ref_dist)
        { 
          ref_dist = tof_rx;
	  for (int k = 0; k < 3; k++)
	  {
            ref_xyz[k] = element_pos[j*3+k] - point_pos[i*3+k];
	  }
        }

      }
      T norm_ref[3];// = new T[Dims];
      for (int k = 0; k < 3; k++)
      {
        norm_ref[k] = point_pos[i*3+k] + ref_xyz[k] / ref_dist;
      }

      //Calc distance to line for all elements

      ref_dist = ref_dist/f_number[0] * 0.5;
      //      if( i == 0)
      //	printf("i: %d, ref_dist: %f\n", i, ref_dist);

      
      //END FIND APOD VALUES
      T v1[3]; //= new T[Dims];
      T v2[3]; //= new T[Dims];
      //T cross[3]; //= new T[Dims];      

      for (int j = 0; j < elements; j++) 
      {
        //T tof_rx = 0;
        //for (int k = 0; k < Dims; k++)
        //{
	// tof_rx += (element_pos[j*Dims+k] - point_pos[i*Dims+k]) * (element_pos[j*Dims+k] - point_pos[i*Dims+k]);
        //}
        //tof_rx = sqrtf(tof_rx);
	T tof_rx = norm3d(element_pos[j*3] - point_pos[i*3], element_pos[j*3+1] - point_pos[i*3+1], element_pos[j*3+2] - point_pos[i*3+2]);
	T tof_rx_inv = (tof_rx+tof_tx+tof_cf)*inv_sof - t_start[0];


	T tof_idx_weight = (tof_rx_inv-t_start_data[0])*f_sampling[0];
	int tof_idx = round(tof_idx_weight);
	tof_idx_weight = tof_idx_weight - tof_idx;
	tof_idx = max(0, min(tof_idx, max_index));
	int tof_idx_plus = tof_idx + 1;
	
        //if(i == 0 && j == 0)
	//  printf("tof_tmp: %e, %d, %d, %e, %e\n", tof_tmp, tof_idx, tof_idx_plus, t_start_data(0), f_sampling(0));

	//BEGIN APOD WEIGHT
        for (int k = 0; k < 3; k++)
        {
          v1[k] = element_pos[j*3+k] - norm_ref[k];
          v2[k] = element_pos[j*3+k] - point_pos[i*3+k];
        }
	//a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x
	T dist_apod = norm3d(v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]); 

	//for( int k = 0; k < Dims; k++)
        //{
        //  int firstIndex = (k+1)%Dims;
        //  int secondIndex = (k+2)%Dims;
	//  cross[k] = v1[firstIndex] * v2[secondIndex] - v1[secondIndex] * v2[firstIndex];
        //}
        //T dist_apod = 0;
        //for( int k = 0; k < Dims; k++)
        //{
        //  dist_apod += cross[k]*cross[k];
        //}
        //dist_apod = sqrt(dist_apod);
        T dist_apod_norm = min((T)dist_apod/ref_dist, (T)1.0);
        T apod_weights=0.5*(1.0+cospi(dist_apod_norm));
	//END APOD WEIGHT

	
	for(int k = 0; k < 16; k++)
	{
	  int k_index = k*sample_size;
	  T sample_idx = samples[k_index + tof_idx*elements+j];
	  T sample_idx_plus = samples[k_index + tof_idx_plus*elements+j];
	  T delayed_sample = sample_idx + (sample_idx_plus - sample_idx) * tof_idx_weight;
	  if(j == 0) bf_samples[i + k*points] = 0;
	  bf_samples[i + k*points] += delayed_sample*apod_weights;
	  //if(i == 0 && j == 0 && k == 0) printf("k %d, samp: %e, w: %e, s: %e, sp: %e, t: %d, tp: %d, index: %d, apod: %e\n", k, delayed_sample, tof_idx_weight,  sample_idx, sample_idx_plus, tof_idx*elements+j, tof_idx_plus*elements+j, k_index + tof_idx*elements+j, apod_weights);

	}
	

	
	
      }

      //delete[] v1;
      //delete[] v2;
      //delete[] cross;      
      //delete[] norm_ref;
      //delete[] ref_xyz;
  }
}

template <typename T>
bool LaunchBeamformingKernel(int elements, int points, int batch_size, int sample_size, int max_index,
			     const T *focus, const T* center_focus, 
			     const T* speed_of_sound, const T* t_start, const T* t_start_data, 
			     const T* f_sampling, const T* f_number, const T* element_pos, 
			     const T* point_pos, const T* samples, T* bf_samples) {
  if (points <= 0) return true;

  constexpr int BLOCK_DIM_X = 32;
;



  BeamformingKernel<T, BLOCK_DIM_X><<<points / BLOCK_DIM_X + (points % BLOCK_DIM_X != 0), BLOCK_DIM_X>>>(elements, points, batch_size, sample_size,
													 max_index, focus, center_focus,
													 speed_of_sound, t_start, t_start_data,
													 f_sampling, f_number, element_pos,
													 point_pos, samples, bf_samples);
  cudaDeviceSynchronize();
  return true;
}

// Explicit instantiations.
template bool LaunchBeamformingKernel<float>(int elements, int points, int batch_size, int sample_size, int max_index, 
					     const float *focus, const float *center_focus, 
					     const float* speed_of_sound, const float* t_start, const float* t_start_data, 
					     const float* f_sampling, const float* f_number, const float* element_pos, 
					     const float* point_pos, const float* samples, float* bf_samples);
template bool LaunchBeamformingKernel<double>(int elements, int points, int batch_size, int sample_size, int max_index, 
					      const double *focus, const double *center_focus, 
					      const double* speed_of_sound, const double* t_start, const double* t_start_data, 
					      const double* f_sampling, const double* f_number, const double* element_pos, 
					      const double* point_pos, const double* samples, double* bf_samples);

#endif
