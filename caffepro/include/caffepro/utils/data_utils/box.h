
#pragma once

#include <conio.h>
#include <algorithm>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"

namespace caffepro {
	namespace data_utils {
		using std::max;
		using std::min;

		// box descriptors
		template <class T>
		struct box_t {
		public:
			// required
			T		left;
			T		top;
			T		right;		// included
			T		bottom;		// included

			// optional
			int		label_id;	// always int
			T		unit_x;		// x-unit
			T		unit_y;		// y-unit

			// methods

			box_t() {}
			box_t(T l, T t, T r, T b, int id = -1, T ux = 1, T uy = 1)
				: left(l), top(t), right(r), bottom(b), label_id(id), unit_x(ux), unit_y(uy) {}
			box_t(const T *v, int id = -1, T ux = 1, T uy = 1)
				: left(v[0]), top(v[1]), right(v[2]), bottom(v[3]), label_id(id), unit_x(ux), unit_y(uy) {}
			box_t(const cv::Rect &rc, int id = -1, T ux = 1, T uy = 1)
				: left(rc.x), top(rc.y), right(rc.x + rc.width - ux), bottom(rc.y + rc.height - uy), label_id(id), unit_x(ux), unit_y(uy) {}

			// available when T == int
			T width()	const { return right - left + unit_x; }

			// available when T == int
			T height()	const { return bottom - top + unit_y; }

			bool valid()	const { return right >= left && bottom >= top; }

			// available when T == int
			T area()	const { return valid() ? width() * height() : 0; }

			box_t<T> intersect(const box_t<T> &other) const {
				box_t<T> result(max(left, other.left), max(top, other.top), min(right, other.right), min(bottom, other.bottom));
				return result.valid() ? result : get_invalid_box();
			}

			double IoU(const box_t<T> &other) const {
				double sI = intersect(other).area();
				double sU = area() + other.area() - sI;
				return sI / sU;
			}

			double IoC(const box_t<T> &other) const {
				double sI = intersect(other).area();
				double sC = other.area();
				return sI / sC;
			}

			box_t<T> flip(const box_t<T> &parent_box) {
				T new_left = parent_box.right - right;
				return box_t<T>(new_left, top, new_left + width() - unit_x, bottom, label_id, unit_x, unit_y);
			}

			void fill(__out T *v)					const { v[0] = left; v[1] = top; v[2] = right; v[3] = bottom; }
			void fill_CX_CY_LOGW_LOGH(__out T *v)	const {
				v[0] = (left + right) / 2; v[1] = (top + bottom) / 2; v[2] = log((float)(right - left)); v[3] = log((float)(bottom - top));
			}

			template <class U>
			void from(const box_t<U> &other) {
				left = (T)other.left; top = (T)other.top; right = (T)other.right; bottom = (T)other.bottom;
				label_id = other.label_id;
				unit_x = (T)other.unit_x; unit_y = (T)other.unit_y;
			}

			// factories
			static box_t<T> get_invalid_box() { return box_t<T>(0, 0, -1, -1, -1); }
			static box_t<T> from_CX_CY_LOGW_LOGH(const T *v) {
				T w = exp((float)v[2]), h = exp((float)v[3]);
				return box_t<T>(v[0] - w / 2, v[1] - h / 2, v[0] + w / 2, v[1] + h / 2);
			}
		};

		typedef box_t<int> Box;
		typedef box_t<float> BoxF;
	}
}