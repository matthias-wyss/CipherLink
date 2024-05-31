import random
import subprocess
import numpy as np
from reedsolo import RSCodec


ALPHABET = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .') # List of characters forming the alphabet.
MESSAGE_LENGTH = 40 # Length of the messages used in the system.
SRV_HOSTNAME = "iscsrv72.epfl.ch" # Hostname of the server used in the system.
SRV_PORT = 80 # Port number of the server used in the system.


def generate_str(size: int = MESSAGE_LENGTH) -> str:
    """
    Generates a random string of specified size.

    Args:
        size (int): The size of the string to generate. By default, uses MESSAGE_LENGTH.

    Returns:
        str: The generated random string.
    """
    
    # Randomly selects 'size' characters from the 'ALPHABET' list
    random_chars = random.sample(ALPHABET, size)
    
    # Concatenates the selected characters into a single string
    random_string = ''.join(random_chars)
    
    # Returns the generated random string
    return random_string


def is_str_in_alphabet(input: str) -> bool:
    """
    Checks if all characters in the input string are part of the alphabet.

    Args:
        input (str): The string to be checked.

    Returns:
        bool: True if all characters are in the alphabet, False otherwise.
    """
    
    # Iterates through each character in the input string
    for char in input:
        # Checks if the character is not in the alphabet
        if char not in ALPHABET:
            # If any character is not in the alphabet, return False
            return False
    # If all characters are in the alphabet, return True
    return True


def compute_energy(vectors: np.ndarray):
    """
    Computes the energy of a set of vectors.

    Args:
        vectors (np.ndarray): An array containing vectors.

    Returns:
        float: The computed energy.
    """
    
    # Compute the square of the absolute values of each element in the array
    squared_abs = np.square(np.abs(vectors))
    
    # Sum all the squared absolute values to get the total energy
    total_energy = np.sum(squared_abs)
    
    # Return the computed energy
    return total_energy


def message_to_bits(message: str, correction_bytes: int = 0) -> np.ndarray:
    """
    Converts a message to binary bits with optional error correction.

    Args:
        message (str): The message to be converted.
        correction_bytes (int): The number of bytes for error correction (default is 0).

    Returns:
        np.ndarray: An array of binary bits.
    """
    
    # Encode the message using the alphabet
    encoded_message = ""
    for char in message:
        char_index = ALPHABET.index(char)
        binary_char = format(char_index, '06b')
        encoded_message += binary_char

    # If error correction is requested, apply Reed-Solomon error correction
    if correction_bytes > 0:
        rs = RSCodec(correction_bytes)
        encoded_bytes = bytes(int(encoded_message[i:i+8], 2) for i in range(0, len(encoded_message), 8))
        encoded_with_correction = rs.encode(encoded_bytes)
        encoded_with_correction_bits = ''.join(format(byte, '08b') for byte in encoded_with_correction)
        encoded_message = encoded_with_correction_bits
    
    # Convert the encoded message to a binary numpy array
    return np.array([int(bit) for bit in encoded_message], dtype=np.int8)


def bits_to_message(bits: np.ndarray, correction_bytes: int = 0) -> str:
    """
    Converts binary bits back to a message with optional error correction.

    Args:
        bits (np.ndarray): An array of binary bits.
        correction_bytes (int): The number of bytes for error correction (default is 0).

    Returns:
        str: The decoded message.
    """
    
    # If error correction is requested, apply Reed-Solomon error correction
    if correction_bytes > 0:
        rs = RSCodec(correction_bytes)
        encoded_bytes = bytearray(int(''.join(map(str, bits[i:i+8])), 2) for i in range(0, len(bits), 8))
        decoded_with_correction = rs.decode(encoded_bytes)[0]
        decoded_with_correction_bits = ''.join(format(byte, '08b') for byte in decoded_with_correction)
        bits = np.array([int(bit) for bit in decoded_with_correction_bits], dtype=np.int8)
    
    # Convert the binary bits back to the original message
    decoded_message = ""
    for i in range(0, len(bits), 6):
        six_bits = bits[i:i+6]
        char_index = int(''.join(map(str, six_bits)), 2)
        decoded_message += ALPHABET[char_index]

    return decoded_message


def channel(x: np.ndarray) -> np.ndarray:
    """
    Simulates a communication channel.

    Args:
        x (np.ndarray): Input vectors.

    Returns:
        np.ndarray: Output vectors after passing through the channel.
    """
    
    n = x.size  # Size of input vectors x
    B = random.choice([0, 1])  # Randomly choose between 0 and 1 for B
    sigma_sq = 25  # Variance of the noise
    
    # Generate Gaussian noise with zero mean and variance sigma_sq
    Z = np.random.normal(0, np.sqrt(sigma_sq), (2 * n))
    
    # Create an array for the transmitted vectors X
    X = np.zeros(2 * n)
    
    # Depending on the value of B, place the input vectors x at the beginning or end of X
    if B == 1:
        X[0:n] = x
    else:
        X[n:2 * n] = x
    
    # Add noise to the transmitted vectors X
    Y = X + Z
    
    # Reshape Y to make it a 1D array
    Y = np.reshape(Y, (-1))
    
    return Y


def get_noisy_vectors_from_server(vectors):
    """
    Retrieves noisy vectors from a server.

    Args:
        vectors: The vectors to be sent to the server.

    Returns:
        np.ndarray: The noisy vectors received from the server.
    """
    
    # Write the vectors to the input file
    with open("input.txt", "w") as input_file:
        for component in vectors:
            input_file.write(str(component) + "\n")

    # Call the Python client script to communicate with the server
    command = ["python3", "client/client.py", "--input_file", "input.txt", "--output_file", "output.txt",
               "--srv_hostname", SRV_HOSTNAME, "--srv_port", str(SRV_PORT)]
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Check if the subprocess executed successfully
    if result.returncode != 0:
        # If server is busy, raise TimeoutError
        if "Last connected < 30s ago. Come back later." in result.stderr:
            raise TimeoutError("Last connected < 30s ago. Come back later.")
        # If vectors energy exceeds the limit, raise ValueError
        elif "Energy of the signal exceeds the limit" in result.stderr:
            energy = compute_energy(vectors)
            raise ValueError(f"Energy of the signal is {energy} and exceeds the limit 40960. Design a more efficient communication system.")
        else:
            raise ValueError(result.stderr)

    # Read the noisy vectors from the output file
    with open("output.txt", "r") as output_file:
        lines = output_file.readlines()
        # Parse the lines into a numpy array of float values for the noisy vectors
        noisy_vectors = np.array([float(val) for line in lines for val in line.strip().split()])

    return noisy_vectors


def detect_useful_part(noisy_vectors: np.ndarray) -> np.ndarray:
    """
    Detects the useful part of noisy vectors by comparing the energies of its two halves.

    Args:
        noisy_vectors: The input noisy vectors.

    Returns:
        The detected useful part of the noisy vectors.
    """
    
    n = len(noisy_vectors) // 2  # Length of each part of noisy_vectors
    
    # Calculate the energy of each part
    energy_1 = compute_energy(noisy_vectors[:n])
    energy_2 = compute_energy(noisy_vectors[n:])
    
    # Check which part has higher energy
    if energy_1 > energy_2:
        useful_noisy_vectors = noisy_vectors[:n]  # First half has higher energy
    else:
        useful_noisy_vectors = noisy_vectors[n:]  # Second half has higher energy
    
    return useful_noisy_vectors


def scale_vectors(vectors: np.ndarray, max_energy: float) -> np.ndarray:
    """
    Scales vectors to match a specified maximum energy.

    Args:
        vectors (np.ndarray): The input vectors.
        max_energy (float): The maximum energy to scale the vectors.

    Returns:
        np.ndarray: The scaled vectors.
    """
    
    # Compute the energy of the input vectors
    energy = compute_energy(vectors)
    
    # Compute the scaling factor to match the maximum energy
    # Subtracting a small value to avoid upper bounding max_energy due to float approximation
    scaling_factor = np.sqrt(max_energy / energy - 0.0000001)
    
    # Scale the vectors using the scaling factor
    return vectors * scaling_factor


def count_errors(message: str, decoded_message: str) -> int:
    """
    Counts the number of errors between the original message and the decoded message.

    Args:
        message (str): The original message.
        decoded_message (str): The decoded message.

    Returns:
        int: The number of errors between the two messages.
    """

    # If the decoded message is None it means that Reed-Solomon decoding failed
    if decoded_message == None:
        return 40

    # Calculate the absolute difference in lengths between the two messages
    count = abs(len(message) - len(decoded_message))

    # Count the number of differing characters at corresponding positions between the two messages
    count += sum(1 for x, y in zip(message, decoded_message) if x != y)

    # Return the total count of errors
    return count
