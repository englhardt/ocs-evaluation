localhost = filter(x -> x[1] == "localhost", worker_list)
remote_servers = filter(x -> x[1] != "localhost", worker_list)

length(localhost) > 0 && addprocs(localhost[1][2], exeflags=exeflags)
length(remote_servers) > 0 && addprocs(remote_servers, sshflags=sshflags, exeflags=exeflags)

# validate package versions
@everywhere function get_git_hash(path)
    cmd = `git -C $path rev-parse HEAD`
    (gethostname(), strip(read(cmd, String)))
end

function setup_julia_environment()
    local_githash = get_git_hash(JULIA_ENV)[2]
    for id in workers()
        remote_name, remote_githash = remotecall_fetch(get_git_hash, id, JULIA_ENV)
        @assert remote_githash == local_githash "Host: $remote_name has version mismatch." *
                                                    "Hash is '$remote_githash' instead of '$local_githash'."
    end
end
@everywhere using Pkg
setup_julia_environment()
