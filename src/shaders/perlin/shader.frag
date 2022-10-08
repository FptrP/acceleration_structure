#version 460 core
const int firstOctave = 3;
const int octaves = 8;
const float persistence = 0.6;


float noise(int x,int y)
{   
  float fx = float(x);
  float fy = float(y);
    
  return 2.0 * fract(sin(dot(vec2(fx, fy) ,vec2(12.9898,78.233))) * 43758.5453) - 1.0;
}

float smoothNoise(int x,int y)
{
  return noise(x,y)/4.0+(noise(x+1,y)+noise(x-1,y)+noise(x,y+1)+noise(x,y-1))/8.0+(noise(x+1,y+1)+noise(x+1,y-1)+noise(x-1,y+1)+noise(x-1,y-1))/16.0;
}

float COSInterpolation(float x,float y,float n)
{
  float r = n*3.1415926;
  float f = (1.0-cos(r))*0.5;
  return x*(1.0-f)+y*f;  
}

float InterpolationNoise(float x, float y)
{
  int ix = int(x);
  int iy = int(y);
  float fracx = x-float(int(x));
  float fracy = y-float(int(y));
    
  float v1 = smoothNoise(ix,iy);
  float v2 = smoothNoise(ix+1,iy);
  float v3 = smoothNoise(ix,iy+1);
  float v4 = smoothNoise(ix+1,iy+1);
    
  float i1 = COSInterpolation(v1,v2,fracx);
  float i2 = COSInterpolation(v3,v4,fracx);
    
  return COSInterpolation(i1,i2,fracy);  
}

float PerlinNoise2D(float x,float y)
{
  float sum = 0.0;
  float frequency =0.0;
  float amplitude = 0.0;
  for(int i=firstOctave;i<octaves + firstOctave;i++)
  {
    frequency = pow(2.0,float(i));
    amplitude = pow(persistence,float(i));
    sum = sum + InterpolationNoise(x*frequency,y*frequency)*amplitude;
  }
    
  return sum;
}

layout (location = 0) in vec2 in_uv;
layout (location = 0) out vec4 color;

void main() {
  float noise = PerlinNoise2D(30.f * in_uv.x, 30.f * in_uv.y);
  color = vec4(noise, noise, noise, noise);
}