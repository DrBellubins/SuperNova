#include <hip/hip_runtime_api.h>
#include <stdio.h>
#define P(x) printf("%4d  %s\n", (int)(x), #x)
int main(){ P(hipDeviceAttributeMaxThreadsPerBlock); P(hipDeviceAttributeMaxSharedMemoryPerBlock);
 P(hipDeviceAttributeMaxRegistersPerBlock); P(hipDeviceAttributeL2CacheSize); P(hipDeviceAttributeMaxThreadsPerMultiProcessor);
 P(hipDeviceAttributeClockRate); P(hipDeviceAttributeMemoryClockRate); P(hipDeviceAttributeMemoryBusWidth);
 P(hipDeviceAttributeMultiprocessorCount); P(hipDeviceAttributePhysicalMultiProcessorCount); P(hipDeviceAttributeWarpSize);
 P(hipDeviceAttributeMaxSharedMemoryPerMultiprocessor); return 0; }
