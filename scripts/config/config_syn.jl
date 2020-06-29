JULIA_ENV = realpath(joinpath(@__DIR__, "..", ".."))
data_root = realpath(joinpath(@__DIR__, "..", "..", "data"))
data_input_root = joinpath(data_root, "input", "synthetic")
data_parameter_file = joinpath(data_input_root, "parameters.jser")
data_output_root = joinpath(data_root, "output")

worker_list = [("localhost", 1)]
exeflags = `--project="$JULIA_ENV"`
sshflags= `-i path/to/ssh/key/file`
SAMPLE_SIZE_LIMIT = 10000

fmt_string = "[{name} | {date} | {level}]: {msg}"
loglevel = "debug"

### data directories ###
data_dirs = ["clusters", "dims", "sizes"]
