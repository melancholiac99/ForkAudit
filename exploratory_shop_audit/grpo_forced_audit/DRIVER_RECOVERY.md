# 2026-11-17 驱动故障与待执行恢复

当前训练停在checkpoint1；128条update2采样已保存，没有丢失已提交更新。
四卡原生logprob接口的NO_SHARD清理兼容错误已修正，但尚未完成四卡实际更新。

内核7.0.0-43-generic、NVIDIA 595.84、HMM开启。15:32:38和15:44:15起，CUDA进程退出阶段出现`uvm_hmm_unregister_gpu`空指针异常（地址0xa0），内核明确输出`Fixing recursive fault but reboot is needed!`。退出后的CUDA进程仍占显存，不能将此状态当成可继续训练。原始日志保存在运行目录`kernel_driver_failure_20261117.log`。

NVIDIA官方仓库同类报告：https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1082 。其中关闭HMM规避了该报告的问题；本机`modinfo nvidia_uvm`也确认支持`uvm_disable_hmm`。这是拟验证的规避，不是已在本机证明修复。

重启获研究者授权后，执行以下已准备操作：

1. 新建`/etc/modprobe.d/forkaudit-nvidia-uvm.conf`，内容`options nvidia_uvm uvm_disable_hmm=1`；保留现有其他配置。
2. 安装本目录三个`forkaudit-shop-*.service`并enable，不在故障内核中启动GPU服务。
3. `sync`后正常重启主机。会短暂中断SSH、TensorBoard及服务器上的其他进程。
4. 开机服务先检查HMM关闭，修复旧boot的GPU账本，运行四卡CUDA/NCCL小测试并检查进程退出后无新内核异常；随后从checkpoint1恢复并复用update2采样。
5. 四卡更新/保存检查通过后自动继续原实验；服务`Restart=no`，新错误停止并保留日志，不循环重采。

研究者已授权应用配置并自动续跑，随后指定由研究者自己重启。modprobe配置、三个开机服务和initramfs已安装/更新，服务已enable但未start；助手未发起重启。恢复后完整四卡吞吐尚需实测；此前预计4–6小时（约30更新平台）或10–14小时（100更新）不包含本次故障修复等待。
