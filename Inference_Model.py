from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import itertools
import threading

import numpy as np
from six.moves import zip  # pylint: disable=redefined-builtin

from google.protobuf import json_format
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import function
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.mixed_precision.experimental import autocast_variable
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
# A module that only depends on `keras.layers` import these from here.
from tensorflow.python.keras.utils.generic_utils import to_snake_case  # pylint: disable=unused-import
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_tensor_list  # pylint: disable=unused-import
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import layer_utils as trackable_layer_utils
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
import tensorflow as tf

class Keras_Model_for_Inference(tf.keras.Model):
    '''
    This class is created to solve an issue that 'tf.Tensor'
    cannot be used in 'tf.cond' or 'tf.while_loop' while inference
    due to an autograph problem. The inference function is a
    clone of __call__ of 'tf.kreas.layers.Layer'. Inference
    is implemented at '_inference'.
    '''
    def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called
        super()

    def inference(self, inputs, *args, **kwargs):

        call_context = base_layer_utils.call_context()
        input_list = nest.flatten(inputs)

        # We will attempt to build a TF graph if & only if all inputs are symbolic.
        # This is always the case in graph mode. It can also be the case in eager
        # mode when all inputs can be traced back to `keras.Input()` (when building
        # models using the functional API).
        build_graph = tf_utils.are_all_symbolic_tensors(input_list)

        # Accept NumPy and scalar inputs by converting to Tensors.
        if any(isinstance(x, (np.ndarray, float, int)) for x in input_list):
            def _convert_non_tensor(x):
                # Don't call `ops.convert_to_tensor` on all `inputs` because
                # `SparseTensors` can't be converted to `Tensor`.
                if isinstance(x, (np.ndarray, float, int)):
                    return ops.convert_to_tensor(x)
                return x
            inputs = nest.map_structure(_convert_non_tensor, inputs)
            input_list = nest.flatten(inputs)

        # Handle `mask` propagation from previous layer to current layer. Masks can
        # be propagated explicitly via the `mask` argument, or implicitly via
        # setting the `_keras_mask` attribute on the inputs to a Layer. Masks passed
        # explicitly take priority.
        mask_arg_passed_by_framework = False
        input_masks = self._collect_input_masks(inputs, args, kwargs)
        if (self._expects_mask_arg and input_masks is not None and
                not self._call_arg_was_passed('mask', args, kwargs)):
            mask_arg_passed_by_framework = True
            kwargs['mask'] = input_masks

        # If `training` argument was not explicitly passed, propagate `training`
        # value from this layer's calling layer.
        training_arg_passed_by_framework = False
        # Priority 1: `training` was explicitly passed.
        if self._call_arg_was_passed('training', args, kwargs):
            training_value = self._get_call_arg_value('training', args, kwargs)
            if not self._expects_training_arg:
                kwargs.pop('training')
        else:
            training_value = None
            # Priority 2: `training` was passed to a parent layer.
            if call_context.training is not None:
                training_value = call_context.training
            # Priority 3a: `learning_phase()` has been set.
            elif backend.global_learning_phase_is_set():
                training_value = backend.learning_phase()
            # Priority 3b: Pass the `learning_phase()` if in the Keras FuncGraph.
            elif build_graph:
                with backend.get_graph().as_default():
                    if base_layer_utils.is_in_keras_graph():
                        training_value = backend.learning_phase()

            if self._expects_training_arg and training_value is not None:
                # Force the training_value to be bool type which matches to the contract
                # for layer/model call args.
                if tensor_util.is_tensor(training_value):
                    training_value = math_ops.cast(training_value, dtypes.bool)
                else:
                    training_value = bool(training_value)
                kwargs['training'] = training_value
                training_arg_passed_by_framework = True

        # Only create Keras history if at least one tensor originates from a
        # `keras.Input`. Otherwise this Layer may be being used outside the Keras
        # framework.
        if build_graph and base_layer_utils.needs_keras_history(inputs):
            base_layer_utils.create_keras_history(inputs)

        # Clear eager losses on top level model call.
        # We are clearing the losses only on the top level model call and not on
        # every layer/model call because layer/model may be reused.
        if (base_layer_utils.is_in_eager_or_tf_function() and
                not call_context.in_call):
            self._clear_losses()

        with call_context.enter(self, inputs, build_graph, training_value):
            # Check input assumptions set after layer building, e.g. input shape.
            if build_graph:
                # Symbolic execution on symbolic tensors. We will attempt to build
                # the corresponding TF subgraph inside `backend.get_graph()`
                # TODO(reedwm): We should assert input compatibility after the inputs
                # are casted, not before.
                input_spec.assert_input_compatibility(self.input_spec, inputs,
                                                                                            self.name)
                if (any(isinstance(x, ragged_tensor.RaggedTensor) for x in input_list)
                        and self._supports_ragged_inputs is False):    # pylint: disable=g-bool-id-comparison
                    raise ValueError('Layer %s does not support RaggedTensors as input. '
                                                     'Inputs received: %s. You can try converting your '
                                                     'input to an uniform tensor.' % (self.name, inputs))

                graph = backend.get_graph()
                with graph.as_default(), backend.name_scope(self._name_scope()):
                    # Build layer if applicable (if the `build` method has been
                    # overridden).
                    self._maybe_build(inputs)
                    cast_inputs = self._maybe_cast_inputs(inputs)

                    # Wrapping `call` function in autograph to allow for dynamic control
                    # flow and control dependencies in call. We are limiting this to
                    # subclassed layers as autograph is strictly needed only for
                    # subclassed layers and models.
                    # tf_convert will respect the value of autograph setting in the
                    # enclosing tf.function, if any.
                    if (base_layer_utils.is_subclassed(self) and
                            not base_layer_utils.from_saved_model(self)):
                        call_fn = autograph.tf_convert(
                                self._inference, ag_ctx.control_status_ctx())
                    else:
                        call_fn = self._inference

                    if not self.dynamic:
                        try:
                            with base_layer_utils.autocast_context_manager(
                                    self._compute_dtype):
                                # Add auto_control_deps in V2 when they are not already added by
                                # a `tf.function`.
                                if (ops.executing_eagerly_outside_functions() and
                                        not base_layer_utils.is_in_eager_or_tf_function()):
                                    with auto_control_deps.AutomaticControlDependencies() as acd:
                                        outputs = call_fn(cast_inputs, *args, **kwargs)
                                        # Wrap Tensors in `outputs` in `tf.identity` to avoid
                                        # circular dependencies.
                                        outputs = base_layer_utils.mark_as_return(outputs, acd)
                                else:
                                    outputs = call_fn(cast_inputs, *args, **kwargs)

                        except errors.OperatorNotAllowedInGraphError as e:
                            raise TypeError('You are attempting to use Python control '
                                                            'flow in a layer that was not declared to be '
                                                            'dynamic. Pass `dynamic=True` to the class '
                                                            'constructor.\nEncountered error:\n"""\n' +
                                                            str(e) + '\n"""')
                    else:
                        # We will use static shape inference to return symbolic tensors
                        # matching the specifications of the layer outputs.
                        # Since `self.dynamic` is True, we will never attempt to
                        # run the underlying TF graph (which is disconnected).
                        # TODO(fchollet): consider py_func as an alternative, which
                        # would enable us to run the underlying graph if needed.
                        outputs = self._symbolic_call(inputs)

                    if outputs is None:
                        raise ValueError('A layer\'s `call` method should return a '
                                                         'Tensor or a list of Tensors, not None '
                                                         '(layer: ' + self.name + ').')
                    if base_layer_utils.have_all_keras_metadata(inputs):
                        if training_arg_passed_by_framework:
                            kwargs.pop('training')
                        if mask_arg_passed_by_framework:
                            kwargs.pop('mask')
                        inputs, outputs = self._set_connectivity_metadata_(
                                inputs, outputs, args, kwargs)
                    self._handle_activity_regularization(inputs, outputs)
                    self._set_mask_metadata(inputs, outputs, input_masks)
                    if hasattr(self, '_set_inputs') and not self.inputs:
                        # Subclassed network: explicitly set metadata normally set by
                        # a call to self._set_inputs().
                        # TODO(b/120997007): This should be done in Eager as well, but
                        # causes garbage collection issues because of the placeholders
                        # created on the default Keras graph.
                        self._set_inputs(inputs, outputs)
            else:
                # Eager execution on data tensors.
                with backend.name_scope(self._name_scope()):
                    self._maybe_build(inputs)
                    cast_inputs = self._maybe_cast_inputs(inputs)
                    with base_layer_utils.autocast_context_manager(
                            self._compute_dtype):
                        outputs = self._inference(cast_inputs, *args, **kwargs)
                    self._handle_activity_regularization(inputs, outputs)
                    self._set_mask_metadata(inputs, outputs, input_masks)

        return outputs

    def _inference(self, inputs, *args, **kwargs):
        raise NotImplementedError

    @trackable.no_automatic_dependency_tracking
    def _init_graph_network(self, inputs, outputs, name=None, **kwargs):
        generic_utils.validate_kwargs(
                kwargs, {'trainable'},
                'Functional models may only specify `name` and `trainable` keyword '
                'arguments during initialization. Got an unexpected argument:')
        # Normalize and set self.inputs, self.outputs.
        if isinstance(inputs, list) and len(nest.flatten(inputs)) == 1:
            inputs = inputs[0]
        if isinstance(outputs, list) and len(nest.flatten(outputs)) == 1:
            outputs = outputs[0]
        self._nested_outputs = outputs
        self._nested_inputs = inputs
        self.inputs = nest.flatten(inputs)
        self.outputs = nest.flatten(outputs)

        if any(not hasattr(tensor, '_keras_history') for tensor in self.outputs):
            base_layer_utils.create_keras_history(self._nested_outputs)

        self._base_init(name=name, **kwargs)
        self._validate_graph_inputs_and_outputs()

        # A Network does not create weights of its own, thus it is already
        # built.
        self.built = True
        self._compute_output_and_mask_jointly = True
        self._is_graph_network = True
        # `_expects_training_arg` is True since the `training` argument is always
        # present in the signature of the `call` method of a graph network.
        self._expects_training_arg = True
        self._expects_mask_arg = True
        # A graph network does not autocast inputs, as its layers will cast them
        # instead.
        self._autocast = False

        self._input_layers = []
        self._output_layers = []
        self._input_coordinates = []
        self._output_coordinates = []

        self._supports_ragged_inputs = None

        # This is for performance optimization when calling the Network on new
        # inputs. Every time the Network is called on a set on input tensors,
        # we compute the output tensors, output masks and output shapes in one pass,
        # then cache them here. When any of these outputs is queried later, we
        # retrieve it from there instead of recomputing it.
        self._output_mask_cache = {}
        self._output_tensor_cache = {}
        self._output_shape_cache = {}

        # Build self._output_layers:
        for x in self.outputs:
            layer, node_index, tensor_index = x._keras_history    # pylint: disable=protected-access
            self._output_layers.append(layer)
            self._output_coordinates.append((layer, node_index, tensor_index))

        # Build self._input_layers:
        for x in self.inputs:
            layer, node_index, tensor_index = x._keras_history    # pylint: disable=protected-access
            # It's supposed to be an input layer, so only one node
            # and one tensor output.
            assert node_index == 0
            assert tensor_index == 0
            self._input_layers.append(layer)
            self._input_coordinates.append((layer, node_index, tensor_index))

        # Keep track of the network's nodes and layers.
        nodes, nodes_by_depth, layers, _ = _map_graph_network(
                self.inputs, self.outputs)
        self._network_nodes = nodes
        self._nodes_by_depth = nodes_by_depth
        self._layers = layers
        self._layer_call_argspecs = {}
        for layer in self._layers:
            self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)
            layer._attribute_sentinel.add_parent(self._attribute_sentinel)

        self._track_layers(layers)

        # Create the node linking internal inputs to internal outputs.
        node_module.Node(
                outbound_layer=self,
                inbound_layers=[],
                node_indices=[],
                tensor_indices=[],
                input_tensors=self._nested_inputs,
                output_tensors=self._nested_outputs)

        # Build self.input_names and self.output_names.
        self._set_output_names()
        self.input_names = []
        self._feed_input_names = []
        self._feed_inputs = []
        self._feed_input_shapes = []
        for layer in self._input_layers:
            self.input_names.append(layer.name)
            if layer.is_placeholder:
                self._feed_input_names.append(layer.name)
                # Use batch_input_shape here because non-eager composite tensors may not
                # have a shape attribute that's meaningful (sparse, for instance, has
                # a tensor that's non-constant and needs to be fed). This means that
                # input layers that create placeholders will need to have the
                # batch_input_shape attr to allow for input shape validation.
                self._feed_input_shapes.append(layer._batch_input_shape)
                self._feed_inputs.append(layer.input)