from typing import Optional, Sequence, Tuple, NamedTuple
from pennylane import numpy as np
import jax.numpy as jnp
import pennylane as qml
import haiku as hk
import jax

class MyGateInitializer(hk.initializers.Initializer):
    def __call__(self, shape: Sequence[int], dtype=int) -> jax.Array:
        gate_idx = jax.random.randint(key=hk.next_rng_key(), shape=[shape[0], shape[1]], minval=0, maxval=3)
        return jax.nn.one_hot(gate_idx, 3)#jnp.ones(shape)
    
class QCNN(hk.Module):
    def __init__(self, kernel_size: Tuple = (2, 2, 3), 
                 n_layers: int = 10, 
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.n_qubits = kernel_size[0] * kernel_size[1]
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.sliced_data_shape = (kernel_size[0]*kernel_size[1], kernel_size[2])

        dev = qml.device("default.qubit.jax", wires=range(self.n_qubits))
        qnode = qml.QNode(self.circuit, dev)
        self.circ = jax.vmap(qnode, (None, None, 0)) # vectorize the kernel applications
        self.vcirc = jax.vmap(self.circ, (None, None, 0)) # vectorize the batches

    def basic_hea(self, gates, angles):
        angles_mod = gates * angles

        for il in range(self.n_layers):
            # rotations
            for iq in range(self.n_qubits):
                qml.RX(angles_mod[il, iq, 0], iq)
                qml.RY(angles_mod[il, iq, 1], iq)
                qml.RZ(angles_mod[il, iq, 2], iq)
            
            for ie in range(1, self.n_qubits, 2): # odd
                qml.CZ(wires=[ie-1, ie])
            
            for ie in range(2, self.n_qubits, 2): # even
                qml.CZ(wires=[ie-1, ie])

    def circuit(self, gates, angles, data):
        for j in range(self.n_qubits):
            qml.Rot(jnp.pi * data[j, 0], 
                    jnp.pi * data[j, 1], 
                    jnp.pi * data[j, 2], 
                    wires=j)

        self.basic_hea(gates, angles)       
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def mod_images(self, images):
        slice = lambda image, i, j: jax.lax.dynamic_slice(image, (i, j, 0), self.kernel_size).reshape(self.sliced_data_shape)
        slice_i = jax.vmap(slice, (None, None, 0))
        slice_ij = jax.vmap(slice_i, (None, 0, None))
        slice_ij_images = jax.vmap(slice_ij, (0, None, None))

        dim_i, dim_j = self._get_target_dims(images[0])
        result = slice_ij_images(images, jnp.arange(dim_i), jnp.arange(dim_j))

        return result.reshape((len(images), -1, ) + self.sliced_data_shape)

    def _get_target_dims(self, image):
        dim_i = len(image)+1-self.kernel_size[0]
        dim_j = len(image[0])+1-self.kernel_size[1]
        return dim_i, dim_j

    def __call__(self, images):
        gates_init = MyGateInitializer()
        angles_init = hk.initializers.RandomUniform(-jnp.pi, jnp.pi)
        gates = hk.get_parameter("gates", shape=[self.n_layers, self.n_qubits, 3], dtype=images.dtype, init=gates_init)
        angles = hk.get_parameter("angles", shape=[self.n_layers, self.n_qubits], dtype=images.dtype, init=angles_init)
        angles_mod = angles.repeat(3).reshape(self.n_layers, self.n_qubits, 3)
        #print(gates)
        image_mod = self.mod_images(images)
        dim_i, dim_j = self._get_target_dims(images[0])
        result = self.vcirc(angles_mod, gates, image_mod).reshape((len(images), dim_i, dim_j, self.n_qubits))

        return result
