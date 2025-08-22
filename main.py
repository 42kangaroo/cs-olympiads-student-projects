# Implements the file to call that takes in the arguments, calls create_image and calls the cpp to save it as jxl

import os
# Suppress XLA/CUDA warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow/XLA logs
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['JAX_LOG_COMPILES'] = '0'  # Suppress JAX compilation logs

import sys
import compression_model_B as cmb
import jax.numpy as jnp
import jax
# Additional JAX logging suppression
jax.config.update('jax_log_compiles', False)
jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')

from PIL import Image
import codex as cdx
import faster_wasserstein_vgg16 as fastW
import subprocess
import logging

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

filename = ""
foldername = ""
lambd = 0
gamma = 0
log2_sigma_value = 0
layers = 2
turns = 100000
patience = turns
delta = 1e-6
take_settings = False
print(sys.argv)


# flag args
if('-l' in sys.argv):
    lambd = float(sys.argv[sys.argv.index('-l') + 1])
    sys.argv.pop(sys.argv.index('-l') + 1)
    sys.argv.pop(sys.argv.index('-l'))

if('-g' in sys.argv):
    gamma = float(sys.argv[sys.argv.index('-g') + 1])
    sys.argv.pop(sys.argv.index('-g') + 1)
    sys.argv.pop(sys.argv.index('-g'))

if('-s' in sys.argv):
    log2_sigma_value = float(sys.argv[sys.argv.index('-s') + 1])
    sys.argv.pop(sys.argv.index('-s') + 1)
    sys.argv.pop(sys.argv.index('-s'))

if('-t' in sys.argv):
    turns = int(sys.argv[sys.argv.index('-t') + 1])
    sys.argv.pop(sys.argv.index('-t') + 1)
    sys.argv.pop(sys.argv.index('-t'))

if('-p' in sys.argv):
    patience = int(sys.argv[sys.argv.index('-p') + 1])
    sys.argv.pop(sys.argv.index('-p') + 1)
    sys.argv.pop(sys.argv.index('-p'))

if('-d' in sys.argv):
    delta = float(sys.argv[sys.argv.index('-d') + 1])
    sys.argv.pop(sys.argv.index('-d') + 1)
    sys.argv.pop(sys.argv.index('-d'))

output_path = ""
if('-o' in sys.argv):
    output_path = sys.argv[sys.argv.index('-o') + 1]
    sys.argv.pop(sys.argv.index('-o') + 1)
    sys.argv.pop(sys.argv.index('-o'))

if('-c' in sys.argv):
    take_settings = True
    sys.argv.pop(sys.argv.index('-c'))

# must be there args
if len(sys.argv) < 2:
    filename = input("Enter the path to the file: ")
elif( "/" == sys.argv[1][-1] ):
    foldername = sys.argv[1]
else:
    filename = sys.argv[1]

if(len(sys.argv) > 2):
    layers = int(sys.argv[2])

# settings
settings = {}
f = open("settings.conf", "r")
for line in f:
    key, value = line.strip().split(": ")
    settings[key] = value
f.close()

# load VGG16 model
cdx.loss.load_vgg16_model()

# Create file list
filelist = []
if foldername != "":
    filelist = [(foldername + f) for f in os.listdir(foldername)]
    if output_path[-1] != "/":
        output_path += "/"
    else:
        output_path += "out/"
else:
    filelist = [filename]
    if output_path == "":
        output_path = "out/" + filename + ".jxl"



print(filelist)
for f in filelist:
    print(bcolors.OKGREEN + f + bcolors.ENDC)
    target = []
    try:
        with Image.open(f, 'r') as file:
            target = jnp.transpose(jnp.asarray(file, dtype=jnp.float32), (2, 0, 1)) / 255.  # CHW
            target = jnp.array(target[:, :(target.shape[1]//16)*16, :(target.shape[2]//16)*16])
            ff = f.replace('/', '_').split('.')[0]

            try:
                target_features = fastW.get_features(target)
                cmb.create_image(
                    target,
                    target_features,
                    lambd * 1.0,
                    gamma,
                    log2_sigma_value * 1.0,
                    turns,
                    layers,
                    ff,
                )
                command = [f'./build/jxl_layered_encoder']
                for i in range(layers):
                    command.append(f'out/reconstructed_image_xyb_{ff}_{i}.txt')
                if take_settings:
                    for key, value in settings.items():
                        command.insert(key+2, value)
                        command.insert(key+2, f"-s")
                
                command.append('-o')
                if(len(filelist) > 1):
                    command.append(output_path + f"{ff}.jxl")
                elif output_path == "":
                    output_path = "out/" + f"{ff}.jxl"
                    command.append(output_path)
                else:
                    command.append(output_path)

                subprocess.run(command, check=True)

            except Exception as e:
                print(bcolors.FAIL + f"Error creating image:" + bcolors.ENDC + f"{e}")
    except FileNotFoundError:
        print(bcolors.FAIL + f"File not found:" + bcolors.ENDC + f"{f}")
    except Exception as e:
        print(bcolors.FAIL + f"Error reading file:" + bcolors.ENDC + f"{e}")