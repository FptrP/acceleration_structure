#ifndef BRDF_GLSL_INCLUDED
#define BRDF_GLSL_INCLUDED

const float PI = 3.1415926535897932384626433832795;

vec3 fresnelSchlick(float cos_theta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}  

vec3 F0_approximation(vec3 albedo, float metallic) {
  vec3 F0 = vec3(0.04);
  return mix(F0, albedo, metallic);
}
#if 0

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
  float a      = roughness*roughness;
  float a2     = a*a;
  float NdotH  = max(dot(N, H), 0.0);
  float NdotH2 = NdotH*NdotH;
	
  float num   = a2;
  float denom = (NdotH2 * (a2 - 1.0) + 1.0);
  denom = PI * denom * denom;
	
  return num / denom;
}
#else

float DistributionGGX(vec3 N, vec3 H, float alpha)
{
  float NoH = dot(N, H);
  float alpha2 = alpha * alpha;
  float NoH2 = NoH * NoH;
  float den = NoH2 * alpha2 + (1 - NoH2);
  return (((NoH2 > 0)? 1 : 0) * alpha2) / ( PI * den * den );
}

#endif

//Heitz2014Microfacet.pdf
float brdfG1(float alpha2, float NdotV) {
  float NdotV2 = NdotV * NdotV;
  float tgv2 = (1 - NdotV2)/NdotV2;
  return 2.0/(1 + sqrt(1 + alpha2 * tgv2)); 
}
//more accurate
float brdfG2(float NdotV, float NdotL, float alpha2) {
  float NdotV2 = NdotV * NdotV;
  float NdotL2 = NdotL * NdotL;

  float L1 = sqrt(1 + alpha2 * (1 - NdotV2)/NdotV2);
  float L2 = sqrt(1 + alpha2 * (1 - NdotL2)/NdotL2);
  return 2.0/(L1 + L2);
}

//Smith-Schlick approximation
float geometryGGX(float NdotV, float NdotL, float alpha2) {
  return brdfG1(alpha2, NdotV) * brdfG1(alpha2, NdotL);
}



float GeometrySchlickGGX(float NdotV, float roughness)
{
  float r = (roughness + 1.0);
  float k = (r*r) / 8.0;

  float num   = NdotV;
  float denom = NdotV * (1.0 - k) + k;
  return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  float ggx2  = GeometrySchlickGGX(NdotV, roughness);
  float ggx1  = GeometrySchlickGGX(NdotL, roughness);
  return ggx1 * ggx2;
}
#if 0  
float sampleGGXdirPDF(in sampler2D PDF_TEX, in vec3 V, in vec3 N, in vec3 L, float roughness) {
  vec3 Y = normalize(cross(V, N));
  vec3 X = normalize(cross(Y, V));

  //cos_theta
  vec3 Lproj = normalize(L -  V * dot(V, L)); 
  float cos_theta = dot(X, Lproj);

  //normal
  const float cos_phin = dot(N, X);
  const float sin_phin = sqrt(1 - cos_phin * cos_phin);

  const float alpha2 = roughness * roughness;

  const float coef = sqrt(1 - alpha2);
  const float p = coef * sin_phin;
  const float q = 0.5 * (coef * cos_phin * cos_theta) + 0.5; //[-1; 1] -> [0; 1]

  float NdotV = max(dot(N, V), 0);
  float pdf = alpha2 * brdfG1(alpha2, NdotV)/(2 * PI * NdotV + 0.00001) * texture(PDF_TEX, vec2(p, q)).x;
  return pdf;
}
#else
float sampleGGXdirPDF(in sampler2D PDF_TEX, in vec3 V, in vec3 N, in vec3 L, float alpha) {
  vec3 Y = normalize(cross(V, N));
  vec3 X = normalize(cross(Y, V));
  alpha = clamp(alpha, 0, 0.9);
  //cos_theta
  vec3 Lproj = normalize(L -  V * dot(V, L)); 
  float cos_theta = dot(X, Lproj);

  //normal
  const float cos_phin = dot(N, X);
  const float sin_phin = sqrt(1 - cos_phin * cos_phin);

  const float alpha2 = alpha * alpha;
  const float coef = sqrt(1 - alpha2);
  const float a = 0.5 * coef * cos_phin * cos_theta + 0.5;
  const float b = coef * sin_phin;

  float pdf = alpha2/(2 * PI * coef) * texture(PDF_TEX, vec2(a, b)).x;
  //return max(pdf, 1.0/(2 * PI));
  //return 1.0/(2 * PI);
  return pdf;
}
#endif

// Input Ve: view direction
// Input alpha_x, alpha_y: roughness parameters
// Input U1, U2: uniform random numbers
// Output Ne: normal sampled with PDF D_Ve(Ne) = G1(Ve) * max(0, dot(Ve, Ne)) * D(Ne) / Ve.z
vec3 sampleGGXVNDF(vec3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
  // Section 3.2: transforming the view direction to the hemisphere configuration
  vec3 Vh = normalize(vec3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
  // Section 4.1: orthonormal basis (with special case if cross product is zero)
  float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
  vec3 T1 = lensq > 0 ? vec3(-Vh.y, Vh.x, 0) * inversesqrt(lensq) : vec3(1,0,0);
  vec3 T2 = cross(Vh, T1);
  // Section 4.2: parameterization of the projected area
  float r = sqrt(U1);
  float phi = 2.0 * PI * U2;
  float t1 = r * cos(phi);
  float t2 = r * sin(phi);
  float s = 0.5 * (1.0 + Vh.z);
  t2 = (1.0 - s)*sqrt(1.0 - t1*t1) + s*t2;
  // Section 4.3: reprojection onto hemisphere
  vec3 Nh = t1*T1 + t2*T2 + sqrt(max(0.0, 1.0 - t1*t1 - t2*t2))*Vh;
  // Section 3.4: transforming the normal back to the ellipsoid configuration
  vec3 Ne = normalize(vec3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.0, Nh.z)));
  return Ne;
}

#endif