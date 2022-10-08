#ifndef CAMERA_HPP_INCLUDED
#define CAMERA_HPP_INCLUDED

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <SDL2/SDL.h>
#include <iostream>

namespace scene {

  const float YAW        =  90.0f;
  const float PITCH      =  0.0f;
  const float SPEED      =  15.0f;
  const float SENSITIVTY =  0.25f;
  const float ZOOM       =  45.0f;

  struct Camera {
	  Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up_ = glm::vec3(0.0f, -1.0f, 0.0f), float yaw_ = YAW, float pitch_ = PITCH) 
	    : pos {position}, world_up {up_}, yaw {yaw_}, pitch {pitch_}
	  {
	    update_camera_vectors();
	  }

	  glm::mat4 get_view_mat() const {
		  return glm::lookAt(pos, pos + front, up);
    }

	  void update_camera_vectors() {
		  glm::vec3 f;
		  f.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
		  f.y = sin(glm::radians(pitch));
		  f.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
		  front = glm::normalize(f);
		  right = glm::normalize(glm::cross(front, this->world_up));
		  up = glm::normalize(glm::cross(right, this->front));
	  }

	  void process_event(const SDL_Event& e) {
	    if (e.type == SDL_KEYDOWN || e.type == SDL_KEYUP) {
		    float val = (e.type == SDL_KEYDOWN)? 1:0;
		    switch (e.key.keysym.sym) {
		      case SDLK_w:
		        move_dir.x = val;
		        break;
		      case SDLK_s:
		        move_dir.x = -val;
		        break;
		      case SDLK_a:
		        move_dir.z = -val;
		        break;
		      case SDLK_d:
		        move_dir.z = val;
		        break;
		      case SDLK_e:
		        move_dir.y = -val;
		        break;
		      case SDLK_q:
		        move_dir.y = val;
		        break;
		      case SDLK_f:
		        if (val) mouse_capture = !mouse_capture;
		        break;
					case SDLK_SPACE:
						if (e.type == SDL_KEYDOWN) {
							std::cout << "CameraPos : " << pos.x << " " << pos.y << " " << pos.z << "\n";
						}
						break;
		    }
	    }

	    if (e.type == SDL_MOUSEMOTION && mouse_capture) {
		    float xm = -e.motion.xrel * mouse_sensitivity, ym = -e.motion.yrel * mouse_sensitivity;
		    yaw += xm;
		    pitch += ym;
		    pitch = (pitch > 89.f)? 89.f : (pitch < -89.f)? -89.f : pitch;
		    update_camera_vectors();
	    } 
	  }

	  void set_speed(float sp) { speed = sp; }

	  void move(float dt) {
	    pos += speed * dt * (move_dir.x * front + move_dir.y * up + move_dir.z * right);
	  }

    glm::vec3 get_pos() const { return pos; }

  private:
	  glm::vec3 pos, front {0, 0, -1}, up, right, world_up;
	  float yaw, pitch;
	  float movement_speed {SPEED}, mouse_sensitivity {SENSITIVTY};
	  glm::vec3 move_dir {0, 0, 0};
	  bool mouse_capture {false};
	  float speed = 1.f;
  };


}

#endif