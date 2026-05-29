#!/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


from __future__ import print_function
import os
import shlex
import sys
import subprocess

REMOTE_TIMEOUT_SECONDS = 30

REMOTE_SCRIPT = """
user=$1
prog_name=$2
own_pid=$$

ps -eo user=,pid=,args= | awk -v user="$user" -v prog_name="$prog_name" -v own_pid="$own_pid" '
{
    proc_user = $1
    pid = $2
    $1 = ""
    $2 = ""
    args = substr($0, 3)
    if (pid != own_pid && proc_user == user && index(args, prog_name) > 0) {
        print pid
    }
}' | while IFS= read -r pid; do
    kill -9 "$pid"
done
"""


def _matching_pids(user, prog_name):
  ps = subprocess.run(["ps", "-eo", "user=,pid=,args="],
                      check=True,
                      stdout=subprocess.PIPE,
                      universal_newlines=True)
  pids = []
  own_pid = str(os.getpid())
  for line in ps.stdout.splitlines():
    parts = line.strip().split(None, 2)
    if len(parts) < 3:
      continue
    proc_user, pid, args = parts
    if pid != own_pid and proc_user == user and prog_name in args:
      pids.append(pid)
  return pids


def _kill_local(user, prog_name):
  pids = _matching_pids(user, prog_name)
  if pids:
    subprocess.run(["kill", "-9"] + pids, check=False)


def _remote_command(user, prog_name):
  return " ".join([
      "sh",
      "-c",
      shlex.quote(REMOTE_SCRIPT),
      "sh",
      shlex.quote(user),
      shlex.quote(prog_name),
      ])


def _host_name(host):
  host = host.strip()
  if ':' in host:
    host = host[:host.index(':')]
  return host


def main(argv):
  if len(argv) != 4:
    print("usage: {} <hostfile> <user> <prog>".format(argv[0]))
    return 1

  host_file = argv[1]
  user = argv[2]
  prog_name = argv[3]
  kill_cmd = _remote_command(user, prog_name)
  print(kill_cmd)

  # Kill program on remote machines
  remote_processes = []
  with open(host_file, "r") as f:
    for host in f:
      host = _host_name(host)
      if not host:
        continue
      print(host)
      remote_processes.append((
          host,
          subprocess.Popen(["ssh", "-oStrictHostKeyChecking=no", host, kill_cmd],
                  shell=False,
                  stdout=subprocess.PIPE,
                  stderr=subprocess.PIPE)))
      print("Done killing")
  for host, proc in remote_processes:
    try:
      _, stderr = proc.communicate(timeout=REMOTE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
      proc.kill()
      _, stderr = proc.communicate()
      print("Timed out killing on {}".format(host), file=sys.stderr)
    if proc.returncode != 0:
      print("Remote kill command failed on {}: {}".format(
          host, stderr.decode("utf-8", "replace").strip()), file=sys.stderr)

  # Kill program on local machine
  _kill_local(user, prog_name)
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
