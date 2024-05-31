import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Avoid warnings from TensorFlow
import tensorflow as tf
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.fec.conv import ConvEncoder, ViterbiDecoder
from sionna.fec.turbo import TurboEncoder, TurboDecoder
from sionna.fec.polar import Polar5GEncoder, Polar5GDecoder
import warnings
warnings.filterwarnings('ignore') # Avoid warnings
import argparse

from utils import generate_str, message_to_bits, bits_to_message, channel
from utils import get_noisy_vectors_from_server, detect_useful_part, scale_vectors
from utils import compute_energy, count_errors, is_str_in_alphabet, MESSAGE_LENGTH

MAX_ENERGY = 40960 # Maximum energy allowed by the system.
MAX_DIMENSION = 500000 # Maximum allowed dimension for vectors.
SUPPORTED_ENCODING_MODES = ["LDPC", "convolutional", "turbo", "polar-sc", "polar-scl"] # List of supported encoding modes in the system.
NOISE_VAR = 25 # Noise variance of the channel.

# Dictionary mapping encoding modes to encoder classes
ENCODER_DECODER_CLASSES = {
    "LDPC": (LDPC5GEncoder, LDPC5GDecoder),
    "convolutional": (ConvEncoder, ViterbiDecoder),
    "turbo": (TurboEncoder, TurboDecoder),
    "polar-sc": (Polar5GEncoder, Polar5GDecoder),
    "polar-scl": (Polar5GEncoder, Polar5GDecoder)
}


def encode_message(message: str, encoding_mode: str = "polar-scl", max_energy: float = MAX_ENERGY,
                   num_correction_bytes: int = 0, rate: float = 1/3, constraint_length: int = 5) -> np.ndarray:
    """
    Encodes a message based on the specified encoding mode.

    Args:
        message (str): The message to be encoded.
        encoding_mode (str): The encoding mode to be used (default is "polar-scl").
        max_energy (float): The maximum energy for scaling vectors (default is MAX_ENERGY).
        num_correction_bytes (int): The number of bytes for error correction (default is 0).
        rate (float): The code rate (default is 1/3).
        constraint_length (int): The constraint length for convolutional or turbo encoding (default is 5).

    Returns:
        np.ndarray: The encoded vectors.
    """
    
    # Check if the encoding mode is supported
    if encoding_mode not in SUPPORTED_ENCODING_MODES:
        raise ValueError(f"Encoding mode {encoding_mode} is not supported. Choose among {SUPPORTED_ENCODING_MODES}")
    
    # Convert the message to binary bits
    bits = message_to_bits(message, num_correction_bytes)
    
    # Convert bits to TensorFlow tensor
    bits_tf = tf.convert_to_tensor(bits, dtype=tf.float32)
    bits_tf = tf.expand_dims(bits_tf, axis=0)
    
    # Get encoder class based on the encoding mode
    encoder_class, _ = ENCODER_DECODER_CLASSES[encoding_mode]
    
    # Create the encoder
    if encoding_mode in ["convolutional", "turbo"]:
        encoder = encoder_class(rate=rate, constraint_length=constraint_length)
    else:
        encoder = encoder_class(k=len(bits), n=int((1/rate) * len(bits)))

    # Encode the bits using the encoder
    encoded_bits_tf = encoder(bits_tf)
    
    # Map the encoded bits to symbols using a constellation
    constellation = Constellation("pam", num_bits_per_symbol=1)
    mapper = Mapper(constellation=constellation)
    vectors_tf = mapper(encoded_bits_tf)
    
    # Convert TensorFlow tensor to numpy array and scale the vectors
    vectors = np.real(vectors_tf.numpy().astype(float)[0])
    vectors = scale_vectors(vectors, max_energy)
    
    # Check if the dimension of vectors exceeds the maximum allowed dimension
    if len(vectors) > MAX_DIMENSION:
        raise ValueError(f"Error: vectors has dimension {len(vectors)} > {MAX_DIMENSION}")
    
    return vectors


def decode_vectors(useful_noisy_vectors: np.ndarray, encoding_mode: str = "polar-scl", num_correction_bytes: int = 0,
                   rate: float = 1/3, constraint_length: int = 5) -> str:
    """
    Decodes noisy vectors based on the specified encoding mode.

    Args:
        useful_noisy_vectors (np.ndarray): The noisy vectors to be decoded.
        encoding_mode (str): The encoding mode used (default is "polar-scl").
        num_correction_bytes (int): The number of bytes for error correction (default is 0).
        rate (float): The code rate (default is 1/3).
        constraint_length (int): The constraint length for convolutional or turbo encoding (default is 5).

    Returns:
        str: The decoded message.
    """
    
    # Check if the encoding mode is supported
    if encoding_mode not in SUPPORTED_ENCODING_MODES:
        raise ValueError(f"Encoding mode {encoding_mode} is not supported. Choose among {SUPPORTED_ENCODING_MODES}")
    
    # Convert the noisy vectors to TensorFlow tensor and handle as complex numbers
    noisy_vectors_tf = tf.convert_to_tensor(useful_noisy_vectors, dtype=tf.float32)
    noisy_vectors_tf = tf.complex(noisy_vectors_tf, tf.zeros_like(noisy_vectors_tf))
    noisy_vectors_tf = tf.expand_dims(noisy_vectors_tf, axis=0)
    
    # Set noise level and define constellation for demapping
    no = tf.constant(NOISE_VAR, dtype=tf.float32)
    constellation = Constellation("pam", num_bits_per_symbol=1)
    demapper = Demapper(demapping_method="app", constellation=constellation)
    
    # Compute the log-likelihood ratio (LLR) of the noisy channel
    llr_ch = demapper([noisy_vectors_tf, no])
    
    # Get the decoder class based on the encoding mode
    encoder_class, decoder_class = ENCODER_DECODER_CLASSES[encoding_mode]
    
    # Create the decoder
    if encoding_mode in ["convolutional", "turbo"]:
        decoder = decoder_class(rate=rate, constraint_length=constraint_length)
    else:
        encoder = encoder_class(k=int(len(useful_noisy_vectors) * rate), n=len(useful_noisy_vectors))
        if "polar" in encoding_mode:
            decoder = decoder_class(encoder, dec_type="SCL" if "scl" in encoding_mode else "SC")
        else:
            decoder = decoder_class(encoder)

    # Decode the noisy channel LLR
    decoded_bits_tf = decoder(llr_ch)
    
    # Convert decoded bits to numpy array and attempt to decode message from bits
    decoded_bits = decoded_bits_tf.numpy().astype(int)[0]
    try:
        decoded_message = bits_to_message(decoded_bits, num_correction_bytes)
    except Exception:
        # If the Reed_solomon decoding failed
        decoded_message = None
    
    return decoded_message


def encode_decode(message: str, use_server: bool = False, encoding_mode: str = "polar-scl", 
                    max_energy: float = MAX_ENERGY, num_correction_bytes: int = 0, 
                    rate: float = 1/3, constraint_length: int = 5, verbose: bool = False) -> bool:
    """
    Encodes, transmits, detects, and decodes a message, and compares the original and decoded messages.

    Args:
        message (str): The message to transmit
        use_server (bool): Flag indicating whether to use a server for transmitting vectors (default is False).
        encoding_mode (str): The encoding mode to be used (default is "polar-scl").
        max_energy (float): The maximum energy for scaling vectors (default is MAX_ENERGY).
        num_correction_bytes (int): The number of bytes for error correction (default is 0).
        rate (float): The code rate (default is 1/3).
        constraint_length (int): The constraint length for convolutional or turbo encoding (default is 5).
        verbose (bool): Flag indicating whether to print the message, vectors, energy and decoded message (default is False).

    Returns:
        bool: True if the original and decoded messages match, False otherwise.
    """

    # Print the message if verbose
    if verbose:
        print(f"Original message: {message}")

    # Encode the message using the specified encoding mode and print if verbose
    vectors = encode_message(message, encoding_mode, max_energy=max_energy, 
                             num_correction_bytes=num_correction_bytes, rate=rate, 
                             constraint_length=constraint_length)
    if verbose:
        print(f"Vectors: {vectors}")

    # Compute the energy of the vectors and print if verbose
    energy = compute_energy(vectors)
    if verbose:
        print(f"Energy: {energy}")

    # If use_server is True, get noisy vectors from the server; otherwise, simulate channel noise
    noisy_vectors = get_noisy_vectors_from_server(vectors) if use_server else channel(vectors)

    # Detect the useful part of the noisy vectors and print if verbose
    useful_noisy_vectors = detect_useful_part(noisy_vectors)
    if verbose:
        print(f"Useful noisy vectors: {useful_noisy_vectors}")

    # Decode the useful noisy vectors using the specified encoding mode and print if verbose
    decoded_message = decode_vectors(useful_noisy_vectors, encoding_mode, 
                                      num_correction_bytes=num_correction_bytes, rate=rate, 
                                      constraint_length=constraint_length)
    if verbose:
        print(f"Decoded message: {decoded_message}")

    # Compare the original message with the decoded message
    result = message == decoded_message

    # Count the number of errors between the original message and the decoded message and print if verbose
    num_errors = count_errors(message, decoded_message)
    if verbose:
        print(f"Number of errors: {num_errors}")
    
    # Print and return the result
    print(result)
    return result


def generate_encode_decode(use_server: bool = False, encoding_mode: str = "polar-scl", 
                            max_energy: float = MAX_ENERGY, num_correction_bytes: int = 0, 
                            rate: float = 1/3, constraint_length: int = 5, verbose: bool = False) -> bool:
    """
    Generates, encodes, transmits, detects, and decodes a message, and compares the original and decoded messages.

    Args:
        use_server (bool): Flag indicating whether to use a server for transmitting vectors (default is False).
        encoding_mode (str): The encoding mode to be used (default is "polar-scl").
        max_energy (float): The maximum energy for scaling vectors (default is MAX_ENERGY).
        num_correction_bytes (int): The number of bytes for error correction (default is 0).
        rate (float): The code rate (default is 1/3).
        constraint_length (int): The constraint length for convolutional or turbo encoding (default is 5).
        verbose (bool): Flag indicating whether to print the message, vectors, energy and decoded message (default is False).

    Returns:
        bool: True if the original and decoded messages match, False otherwise.
    """

    # Generate a random message
    message = generate_str()
    
    result = encode_decode(message, use_server, encoding_mode, max_energy,
                           num_correction_bytes, rate, constraint_length, verbose)

    return result


def exam_encode_decode(message: str) -> bool:
    """
    Encodes, transmits, detects and decodes a message with chosen parameters for the exam presentation.

    Args:
        message (str): The message to transmit

    Returns:
        bool: True if the original and decoded messages match, False otherwise.
    """
    result = encode_decode(message, use_server=True, encoding_mode="polar-scl",
                           max_energy=0.4*MAX_ENERGY, num_correction_bytes=0,
                           rate=1/4.5, verbose=True)
    return result


def main():
    """
    Parse the input message and call the exam_encode_decode function.
    The message must be 40 character long and within the ALPHABET.
    By default the message is random.
    """
    parser = argparse.ArgumentParser(description='Encodes some message, passes it through a noisy channel and decodes it.')
    parser.add_argument('-m', '--message', type=str, default=generate_str(),
                        help='The message to send must be 40 characters long and within the ALPHABET')
    args = parser.parse_args()
    message = args.message
    
    # Check if the message contains valid characters and has the correct length.
    # If not, raise a ValueError with an appropriate message.
    if not is_str_in_alphabet(message):
        raise ValueError(f"The message {message} contains invalid characters")
    if len(message) != MESSAGE_LENGTH:
        raise ValueError(f"The message length is not {MESSAGE_LENGTH}")
    
    # Call the function to process the validated message.
    exam_encode_decode(message)

    
if __name__ == "__main__":
    main()
    