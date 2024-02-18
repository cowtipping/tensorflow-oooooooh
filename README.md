# Hello, TensorFlow

Playing with ML things.

## What Are All These Files?

```python
tensortest.py           # Checks if a GPU is available.
main.py                 # Builds a model from the MNIST dataset and saves it as mymodel.keras.
model_prediction.py     # Reads the model and saves the model's prediction data to predictions.png and predictions.json.
docker_build.bat        # For the lazy. Double click to build the Docker file (you will still need to manually attach the terminal shell - see below)
```

### Docker Things

#### Why Docker?

> "Docker is the easiest way to enable TensorFlow GPU support on Linux since only the NVIDIA® GPU driver is required on the host machine (the NVIDIA® CUDA® Toolkit does not need to be installed)."  

[TensorFlow docs.](https://www.tensorflow.org/overview)

Anything above version 2.10 of TensorFlow is not supported on the GPU on Windows Native. So you'd have to do `pip install "tensorflow<2.11"`  
The TensorFlow Docker image is a pre-configured environment that includes **TensorFlow**, **Python**, and all the necessary **dependencies** for TensorFlow. It's built on top of a **Linux** base image, which means it runs a Linux environment, even when it's run on a Windows host. As for GPU support, TensorFlow relies on CUDA and cuDNN, which are NVIDIA's GPU-accelerated libraries for deep learning. These libraries are better supported on Linux than on Windows.  
In addition, the TensorFlow Docker image includes the necessary configurations to enable GPU support, so **you don't need to install and configure CUDA and cuDNN manually**.

#### What's Going On Here, Then?

Your Python script and any other files in your workspace are copied into the Docker image, and when you run a Docker container from the image, your script runs in this container. If you have other Python packages that your script depends on, you can add them to a requirements.txt file.

The volumes directive in the `docker-compose.yml` file tells Docker to mount your current directory (.) to the `/app` directory in the Docker container. This allows your Python script to access any files in your current directory.

#### How To Run It

- Load Docker Desktop.
- Build and run with `docker-compose up --build`
- Or if it's already built, just run it `docker-compose up`
- Or just use the Docker extension in VS Code.

To run your Python script inside the Docker container, you need to attach a shell to the running container and run your script in that shell. Here's how you can do it:
- Enter this into a terminal: `docker-compose exec app bash`. Or do the following:
- Open the Docker view in Visual Studio Code by clicking on the Docker icon in the Activity Bar.
- In the Docker view, you should see your running container. Right-click on the container and select "Attach Shell". This will open a new terminal connected to the Docker container.
- In the attached shell, navigate to the directory where your Python script is located (`/app`).
- Run your Python script in the attached shell with the command `python your_script.py`, replacing your_script.py with the name of your Python script.
- This will run your Python script inside the Docker container, where the tensorflow module is available.

Build the model:  
`python main.py`

Check how it's performing:  
`python visual.py`

Gaze at `output.png`.  
Be amazed by numbers.

#### Explanation of docker-compose

The command `/bin/bash -c "tail -f /dev/null"` is often used in Docker and Docker Compose to start a container that does not stop immediately after it's started. This is useful when you want to start a container and keep it running indefinitely for debugging or other purposes. The `tail -f /dev/null` command does essentially nothing: it continuously follows (-f) the end of the /dev/null file, which is a special file that discards all data written to it and provides no data when read from. Since the end of this file never changes, `tail -f /dev/null` runs indefinitely until it's explicitly stopped. If you replace this command with `["python", "/app/script.py"]` to run your Python script, the container will stop as soon as your Python script finishes executing.

The `--no-cache-dir` option disables the pip cache. This can make the image smaller, because it doesn't store the intermediate files that pip uses when it installs packages. It can also ensure that you're always installing the latest version of a package, because pip doesn't use cached older versions.

### Understanding The Code in main.py

The code is implementing a simple neural network to recognize handwritten digits using the **MNIST** dataset, a classic dataset in machine learning, which includes images of handwritten digits (0 through 9). Here's a breakdown of each part of the code:

1. **Loading Data**: 
    - `(x_train, y_train), (x_test, y_test) = mnist.load_data()`
    - This line loads the MNIST dataset. `x_train` and `x_test` contain grayscale image data of handwritten digits, while `y_train` and `y_test` contain the corresponding labels (0 through 9).

2. **Preprocessing Data**: 
    - `x_train = x_train / 255.0`, `x_test = x_test / 255.0`
    - The pixel values in the images are normalized to be between 0 and 1. This helps in the training process by ensuring that the input features are on a similar scale.

3. **Building the Neural Network**:
    - `model = Sequential([Flatten(input_shape=(28, 28)), Dense(128, activation="relu"), Dense(10, activation="softmax")])`
    - Here, you're defining a Sequential model with three layers:
        - `Flatten`: Flattens the 28x28 pixel images into a 1D array of 784 pixels.
        - `Dense(128, activation="relu")`: A dense layer with 128 neurons and ReLU (Rectified Linear Unit) activation function.
        - `Dense(10, activation="softmax")`: The output layer with 10 neurons (one for each digit), using a softmax function to output a probability distribution over the 10 digit classes.

4. **Compiling the Model**: 
    - `model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])`
    - The model is compiled with the Adam optimizer, the sparse categorical crossentropy loss function (suitable for multi-class classification tasks), and the metric to track is accuracy.

5. **Training the Model**: 
    - `model.fit(x_train, y_train, epochs=5)`
    - This line trains the model on the training data for 5 epochs (iterations over the entire dataset).

6. **Evaluating the Model**: 
    - `test_loss, test_acc = model.evaluate(x_test, y_test)`
    - After training, the model's performance is evaluated on the test dataset to see how well it generalizes to new, unseen data.

### What's Next?

1. **Experiment with the Model**:
    - Try changing the number of neurons in the Dense layer or the number of epochs.
    - Experiment with different activation functions like `sigmoid` or `tanh`.

2. **Visualize the Results**:
    - You can plot some of the images along with their predicted labels to see how well your model is performing.

3. **Try a Different Dataset**:
    - Once comfortable with MNIST, you can try using a different dataset and adapt the neural network accordingly.

4. **Learn More About Neural Networks and Deep Learning**:
    - Explore other types of neural networks like Convolutional Neural Networks (CNNs), which are particularly good for image data.

5. **Implement Model Saving and Loading**:
    - Learn how to save a trained model and load it later for prediction.

6. **Explore Advanced Topics**:
    - Once you have a good grasp of the basics, consider exploring more advanced topics in machine learning and deep learning, such as data augmentation, regularisation techniques, or more complex architectures.

Feel free to ask if you want to explore any of these next steps in more detail or if you have specific questions about any other aspect of machine learning!

### TensorFlow GPU vs CPU

As for a fallback to CPU if a user doesn't have the necessary GPU drivers, TensorFlow does this automatically. If you install tensorflow-gpu and TensorFlow cannot access a GPU, it will automatically use the CPU for computations. However, this fallback behavior can be inefficient because the GPU-enabled version of TensorFlow includes extra software that is not used when running on a CPU.

If you want to provide a CPU-only version of your Docker image, you could create a separate Dockerfile that installs the CPU-only version of TensorFlow (tensorflow instead of tensorflow-gpu). Users could then choose the appropriate Dockerfile based on their hardware.

Alternatively, you could check for GPU availability at runtime and install the appropriate version of TensorFlow dynamically. However, this would make your Dockerfile and application more complex, and it's generally recommended to keep Dockerfiles simple and deterministic.

A project structure with two Dockerfiles typically separates the configurations for different environments or requirements. In your case, you might have one Dockerfile for a GPU-enabled environment and another for a CPU-only environment. Here's an example of what this might look like:
```
/my_project
    /app
        app.py
        requirements.txt
    /docker
        Dockerfile.cpu
        Dockerfile.gpu
    README.md
```

In this structure:

- The app directory contains your Python application and its dependencies.
- The docker directory contains the Dockerfiles for different environments.
- Dockerfile.cpu is the Dockerfile for a CPU-only environment. It installs the CPU version of TensorFlow.
- Dockerfile.gpu is the Dockerfile for a GPU-enabled environment. It installs the GPU version of TensorFlow.  

You can build the Docker images using the appropriate Dockerfile with a command like this:  
`docker build -f docker/Dockerfile.cpu -t my_project_cpu .`  
`docker build -f docker/Dockerfile.gpu -t my_project_gpu .`  
This allows you to maintain separate configurations for different environments, while keeping your application code the same. Users can then choose the appropriate Docker image based on their hardware.

### Pre-Docker Notes From Initial Driver Setup on Windows 11

This was the driver setup to get TensorFlow working with CPU and **pip** installations of TensorFlow and chums. It is possible to get GPU support working with a setup involving **WSL 2**, so this is kept here for posterity in case Docker blows up and that route needs to be explored. Scroll down the [TF docs](https://www.tensorflow.org/install/pip) for WSL 2 installation notes.

Install:

- https://www.nvidia.com/drivers
- https://developer.nvidia.com/cuda-toolkit-archive
- https://developer.nvidia.com/cudnn (CUDA Deep Neural Network library)

Drag the following folders from cuDNN into the CUDA directory, typically at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y`

- bin
- include
- lib64 (or lib)

Add paths to enviornment variables (default install locations. Change depending on version number):

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin  
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp  
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\extras\CUPTI\lib64  

`export PATH=$PATH:"/path/to/your/directory"`
