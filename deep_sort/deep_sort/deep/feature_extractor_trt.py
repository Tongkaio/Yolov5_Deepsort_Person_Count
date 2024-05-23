"""
    This file is a TensorRT version modified from `feature_extractor_trt.py`.
"""
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
INPUT_W = 128
INPUT_H = 64


class TrackerExtractor:

    def __init__(self, engine_file_path):
        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1
        self.size = (64, 128)

        # Create a Context on this device,
        # Note: https://documen.tician.de/pycuda/driver.html#pycuda.driver.Device.make_context
        # `make_context`` will create a context, and also 
        # make the newly-created context the current context
        # which means that there is a `Context.push()` in this action
        # so you should do `Context.pop()` for this `push` action
        # two choices:
        # 1. do it at the last line of `make_context`'s code block  
        # 2. do it in the end
        # I choose the first way here
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            dims = engine.get_binding_shape(binding)
            if dims[0] < 0:
                size *= -1
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

        # Remove the context pushed by `cuda.Device(0).make_context()`
        self.cfx.pop()

    def _preprocess(self, im_crops):
        """
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            im_resize = cv2.resize(im.astype(np.float32)/255., size)
            return im_resize
        def _normalize(im):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            return (im.astype(np.float32) - np.array(mean)) / np.array(std)
        imgs = []
        for im in im_crops:     
            img = _normalize(_resize(im, self.size))
            # img = img.cpu().numpy()
            imgs.append(img)
        return imgs

    def track_extractor(self, im_crops):
        # threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        im_batchs = self._preprocess(im_crops)
        features_trt = []
        for im_batch in im_batchs:
            # Copy input image to host buffer
            np.copyto(host_inputs[0], im_batch.ravel())
            # Transfer input data  to the GPU.
            cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
            # Run inference.
            context.set_binding_shape(0, (1, 3, 128, 64))
            context.execute_async(bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
            # Synchronize the stream
            stream.synchronize()
            # Here we use the first row of output in that batch_size = 1
            trt_outputs = host_outputs[0]
            # Do postprocess
            feature_trt = trt_outputs
            features_trt.append(feature_trt)
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        return np.array(features_trt)


if __name__ == '__main__':
    imgs = np.random.randint(0, 255, size=(2, 100, 100, 3))  # simulate a batch(size=2) of imgs
    print(f"imgs.shape: {imgs.shape}")
    extr = TrackerExtractor("checkpoint/deepsort.engine")
    feature = extr.track_extractor(imgs)
    print(f"feature.shape: {feature.shape}")
    # extr.destroy()  # pop the context pushed in `make_context` action
