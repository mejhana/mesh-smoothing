# Mesh Smoothing using iterative implicit smoothing

In this project we smooth the mesh by finding the laplace beltrami operator and perform implicit smoothing. 

## Running the code
1. Open the project directory
1. run `poetry install` and `poetry shell` to set up the environment and install dependencies. 
1. upload your mesh in `meshes` folder and change the location of the mesh in `__main__.py`
1. run the `__main__.py` file
1. You'll see a pop-up window of the mesh being smoothed and the results are stored in `results` folder. 