# Principle of Digital Communications Project code

This code was created as part of a project for the "Principle of Digital Communications" course, taught to third-year students in the Communication Systems section at EPFL. The objective was to encode a 40-character message from an alphabet of 64 possible characters, send it through a noisy channel, and then decode it without errors, all while using the least amount of energy possible.

Group 6: 
- Matthias Wyss (SCIPER 329884)
- Sofia Taouhid (SCIPER 339880)
- Guillaume Vitalis (SCIPER 339432)
- Alexandre Huou (SCIPER 342227)

Our code is based on the python libraries [sionna](https://github.com/NVlabs/sionna) and [reedsolo](https://github.com/tomerfiliba-org/reedsolomon).


## Installation

1. Make sure to have Python 3.8 installed on your computer
2. Create a python's environment:
   ```sh
   python3.8 -m venv pdc_project_code
   ```
3. Activate the environment:
   - On Windows:
     ```sh
     pdc_project_code\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source pdc_project_code/bin/activate
     ```
4. Install the dependencies (make sure to follow the provided versions requirements (especially for Tensorflow)):
   ```sh
   pip install -r requirements.txt
   ```
Note: You may also have to install the `llvm` library on your computer.
   - On macOS:
      ```sh
      brew install llvm
      ```
   - On Windows:
      Dowload [llvm](https://releases.llvm.org) and install it.
      You may also have to modify the variable environment to the corresponding installing path (follow the instructions on the terminal)
## Usage
   You can run our code with a randomly generated message with:
   ```sh
   python design.py
   ```
   You can also define a message directly by using the command -m or --message:
   ```sh
   python design.py -m "This sentence has exactly 40 characters."
   ```
   ```sh
   python design.py --message "My pet rock passed away. Need a new one."
   ```

   By default, when you run `design.py`, you will use our best settings for this particular project. It uses polar codes with a successive cancellation list and a rate of 1/4.5, without the Reed-Solomon layer, with an energy usage of 16384 units.

   But we also implemented the following error correction codes:
   - Low-Density Parity-Check (LDPC)
   - Convolutional Codes
   - Turbo Codes
   - Polar Codes with successive cancellation decoder (Polar-SC)
   - Polar Codes with successive cancellation list decoder (Polar-SCL)

   There is also an option to add a layer of code correction using Reed-Solomon.
