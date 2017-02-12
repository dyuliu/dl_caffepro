
#pragma once

#include <caffepro/layers/cluster_loss_layer.h>

namespace caffepro {
	class online_kmeans_loss_layer : public cluster_loss_layer {
	public:
		online_kmeans_loss_layer(caffepro_context *context, const LayerParameter &param);

	public:
		// overrides
		virtual void init();
		virtual void resize();

	protected:
		// overrides
		virtual void on_forward(int device_index);
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff);
		virtual void on_after_backward();

	protected:
		void init_centers();
		void find_nearest(int device_index);
		void minibatch_kmeans(int device_index);

	protected:
		int update_interval_, update_iters_;
		int current_iter_, current_kmeans_batch_;

		boost::shared_ptr<node_blob> prepare_centers_, prepare_distance_matrix_, prepare_assign_matrix_;
		std::vector<int> center_count_;
	};
}