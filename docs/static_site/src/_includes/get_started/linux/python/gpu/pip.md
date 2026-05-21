**Current fork note:** Apache MXNet upstream was archived on 2023-11-17. The
historical `mxnet-cu*` commands below describe Apache-era packages, not the
current `smolix/mxnet` CUDA 13 preview wheel.

For this fork's Linux x86_64 CPython 3.11 CUDA 13 wheel, install from the
GitHub release:

{% highlight bash %}
pip install https://github.com/smolix/mxnet/releases/download/v2.0.0.cu13.bw.20260517-beta/mxnet-2.0.0+cu13.bw.20260517-cp311-cp311-linux_x86_64.whl
{% endhighlight %}

This wheel requires an NVIDIA driver supporting CUDA 13 (R570+) and a CUDA
13 toolkit at `/usr/local/cuda`. `pip` installs `nvidia-cudnn-cu13` and
`nvidia-nccl-cu13`; `libcudart`, `libcublas`, `libcufft`, `libcusolver`,
`libcurand`, and `libnvrtc` come from the system CUDA toolkit because NVIDIA's
other `nvidia-*-cu13` packages are still placeholder stubs.

The wheel is built with `USE_OPENCV=OFF`: it has no `opencv-python` dependency,
does not link `libopencv_*`, and does not require `libopencv-dev`. Native MXNet
image-decode and RecordIO image helpers that need OpenCV require a source build
with `USE_OPENCV=ON`.

Historical Apache MXNet package guidance follows.

**WARNING**: the following PyPI package names are provided for your convenience but
they point to packages that are *not* provided nor endorsed by the Apache
Software Foundation. As such, they might contain software components with more
restrictive licenses than the Apache License and you'll need to decide whether
they are appropriate for your usage. The packages linked here contain
proprietary parts of the NVidia CUDA SDK and GPL GCC Runtime Library components.
Like all Apache Releases, the official Apache MXNet releases
consist of source code only and are found at the [Download
page](https://mxnet.apache.org/get_started/download).

**PREREQUISITES**: [CUDA](https://developer.nvidia.com/cuda-downloads) should be installed first. Starting from version 1.8.0, [CUDNN](https://developer.nvidia.com/cudnn) and [NCCL](https://developer.nvidia.com/nccl) should be installed as well.

Run the following command:

<div class="v1-9-1">
{% highlight bash %}
$ pip install mxnet-cu102
{% endhighlight %}

</div> <!-- End of v1-9-1 -->

<div class="v1-8-0">
{% highlight bash %}
$ pip install mxnet-cu102==1.8.0.post0
{% endhighlight %}

</div> <!-- End of v1-8-0 -->

<div class="v1-7-0">
{% highlight bash %}
$ pip install mxnet-cu102==1.7.0
{% endhighlight %}

</div> <!-- End of v1-7-0 -->

<div class="v1-6-0">
{% highlight bash %}
$ pip install mxnet-cu102==1.6.0.post0
{% endhighlight %}

</div> <!-- End of v1-6-0 -->

<div class="v1-5-1">
{% highlight bash %}
$ pip install mxnet-cu101==1.5.1
{% endhighlight %}

</div> <!-- End of v1-5-1 -->
<div class="v1-4-1">

{% highlight bash %}
$ pip install mxnet-cu101==1.4.1
{% endhighlight %}

</div> <!-- End of v1-4-1 -->
<div class="v1-3-1">

{% highlight bash %}
$ pip install mxnet-cu92==1.3.1
{% endhighlight %}

</div> <!-- End of v1-3-1-->
<div class="v1-2-1">

{% highlight bash %}
$ pip install mxnet-cu92==1.2.1
{% endhighlight %}

</div> <!-- End of v1-2-1-->

<div class="v1-1-0">

{% highlight bash %}
$ pip install mxnet-cu91==1.1.0
{% endhighlight %}

</div> <!-- End of v1-1-0-->

<div class="v1-0-0">

{% highlight bash %}
$ pip install mxnet-cu90==1.0.0
{% endhighlight %}

</div> <!-- End of v1-0-0-->

<div class="v0-12-1">

{% highlight bash %}
$ pip install mxnet-cu90==0.12.1
{% endhighlight %}

</div> <!-- End of v0-12-1-->

<div class="v0-11-0">

{% highlight bash %}
$ pip install mxnet-cu80==0.11.0
{% endhighlight %}

</div> <!-- End of v0-11-0-->

<br>

{% include /get_started/pip_snippet.md %}
{% include /get_started/gpu_snippet.md %}
