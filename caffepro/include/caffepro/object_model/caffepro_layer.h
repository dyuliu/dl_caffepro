
#pragma once

#include <caffepro/object_model/node_blob.h>
#include <caffepro/object_model/caffepro_config.h>

namespace caffepro {
	
	class LayerParameter;
	
	class caffepro_layer : public caffepro_object {
	public:
		// definations
		typedef std::vector<boost::shared_ptr<node_blob> > layer_io_buffer, layer_weights;
		typedef unsigned int weight_selector, act_selector;

		struct layer_attribute {
			int num_inputs_min, num_inputs_max, num_outputs_min, num_outputs_max;

			enum layer_usage_type : unsigned int {
				USAGE_DEFAULT		= 0U,
				USAGE_DATA_SOURCE	= 1U,
				USAGE_LOSS			= 2U
			};
			layer_usage_type usage;

			enum constraint_flag : unsigned int {
				CF_REQUIRE_SAME_DEVICE							= 1U << 0,
				CF_REQUIRE_UNIQUE_DEVICE						= 1U << 1,
				CF_REQUIRE_SAME_SHAPE							= 1U << 2,
				CF_REQUIRE_SAME_COUNT							= 1U << 3,
				CF_REQUIRE_SAME_NUM								= 1U << 4,
				CF_REQUIRE_SAME_DIMTYPE							= 1U << 5,
				CF_REQUIRE_SAME_NDIM							= 1U << 6,
				CF_REQUIRE_SAME_INNER_COUNT						= 1U << 7,
				CF_REQUIRE_4D									= 1U << 8,
				CF_REQUIRE_FIXEDLEN_DIM							= 1U << 9,
				CF_REQUIRE_NDIM_4								= 1U << 10,
				CF_REQUIRE_SAME_DIMTYPE_ACROSS_DEVICES			= 1U << 11,
				CF_REQUIRE_SAME_NDIM_ACROSS_DEVICES				= 1U << 12,
				CF_REQUIRE_SAME_COUNT_ACROSS_DEVICES			= 1U << 13,
				CF_REQUIRE_SAME_NUM_ACROSS_DEVICES				= 1U << 14,
				CF_REQUIRE_SAME_INNER_COUNT_ACROSS_DEVICES		= 1U << 15,
				CF_ALLOW_INPLACE								= 1U << 16,
				CF_FORBID_INPLACE_USAGE_PREV_WHEN_INPLACE		= 1U << 17,
				CF_FORBID_INPLACE_USAGE_NEXT_WHEN_INPLACE		= 1U << 18,
				CF_FORBID_INPLACE_USAGE_PREV_ALWAYS				= 1U << 19,
				CF_FORBID_INPLACE_USAGE_NEXT_ALWAYS				= 1U << 20
			}; 
			unsigned int constraints;
			unsigned int get_constraint(unsigned int mask) const { return constraints & mask; }
			void set_constraint(unsigned int mask) { constraints |= mask; }
			void clear_constraint(unsigned int mask) { constraints &= ~mask; }

			enum dispatcher_type { INPUT_BASE, OUTPUT_BASE } device_dispatcher_forward, device_dispatcher_backward;
		};

	public:
		caffepro_layer(caffepro_context *context, const LayerParameter &param);
		~caffepro_layer();

	public:
		// fetch functions
		caffepro_context *context() const { return context_; }
		const LayerParameter& layer_param() const { return layer_param_; }
		layer_io_buffer& inputs() { return inputs_; }
		layer_io_buffer& outputs() { return outputs_; }
		layer_weights& weights() { return weights_; }
		layer_weights& internal_weights() { return internal_weights_; }
		const layer_attribute& attr() { return attr_; }
		bool inplace() { return !inputs_.empty() && !outputs_.empty() && inputs_.front() == outputs_.front(); }
		caffepro_config_reader &config() { return config_; }
		const caffepro_config_reader &config() const { return config_; }
		void set_namespace(const std::string &ns) { namespace_ = ns; }
		const std::string& get_namespace() const { return namespace_; }

	public:
		// interfaces
		virtual void bind(layer_io_buffer &inputs, layer_io_buffer &outputs);
		virtual void init();
		virtual void resize();
		virtual void forward();
		virtual void backward(act_selector bp_acts = UINT32_MAX, weight_selector bp_weights = UINT32_MAX, 
			act_selector clear_acts_diff = UINT32_MAX, weight_selector clear_weights_diff = UINT32_MAX);
		virtual void release_all();
		virtual void write_to_proto(LayerParameter *proto);

	protected:
		virtual void check_input();

	protected:
		// overrides
		virtual void on_before_forward() {}
		virtual void on_forward(int device_index) = 0;
		virtual void on_after_forward() {}
		virtual void on_before_backward() {}
		virtual void on_backward(int device_index, act_selector bp_acts, weight_selector bp_weights, act_selector clear_acts_diff, weight_selector clear_weights_diff) = 0;
		virtual void on_after_backward() {}

	public:
		// factory functions
		static caffepro_layer* create(caffepro_context *context, const LayerParameter &param);

	protected:
		caffepro_context *context_;
		const LayerParameter &layer_param_;

		layer_io_buffer inputs_, outputs_;
		layer_weights weights_;
		layer_weights internal_weights_;
		
		layer_attribute attr_;
		caffepro_config_reader config_;
		std::string namespace_;

	private:
		DISABLE_COPY_AND_ASSIGN(caffepro_layer);
	};
}