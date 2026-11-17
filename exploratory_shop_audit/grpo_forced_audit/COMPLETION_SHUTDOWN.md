# 完成后关机

研究者要求整轮实验完成后关机，服务器为智星云KVM。completion_shutdown.py只实现系统内关机检查，不调用平台退租或删除磁盘接口，不承诺关机即停止计费。平台结束租用、保留磁盘属于不同操作，需平台访问权限；当前尚未获得。

研究者现已明确选择：整轮实验完成后，结束智星云租用并保留全部磁盘。该选择已记录在运行目录completion_shutdown/provider_request.json；无需再次询问是否退租。当前未发现平台API凭据，实例身份及平台磁盘保留参数也尚未核验，因此平台自动退租未启用。需要提供API凭据的本地文件路径与控制台实例ID后才能接入；密钥不得进入Git或日志。

现有completion_shutdown.py只提供完成检查和另行授权的系统关机路径，不能执行平台退租。只有只读检查已执行；未写入系统关机授权request.json，未安装/启用执行定时器，不会以系统关机代替平台退租。

守卫要求campaign_status=complete，终点更新数与metrics一致，基础/终点各200题×(1贪心+4采样)完整且无重复，E0/E1自然任务及每个非终止候选的A/B库各3次齐全，最终模型/优化器/各rank状态与dataloader保存完整，rank指纹一致且适配器哈希相符，报告已落盘，没有活动GPU租约或运行中的训练/修复服务。科学判定为证据不足也可正常完成；运行失败或产物缺失不会触发。

只读检查：`python -m exploratory_shop_audit.grpo_forced_audit.completion_shutdown`。执行需要本次boot专属的研究者授权记录且显式传`--execute`。检查通过后先保存回执并sync，再请求`shutdown -h +2`；不重复下达，不因日后重启再次关闭机器。取消已排定的系统关机可用`shutdown -c`。

此检查证明本机产物已保存，不等于异地备份或GitHub发布；不会上传权重或原始轨迹。平台保留磁盘设置、停费效果和云端备份需要另外核实。

平台来源：[智星云数据保留说明](https://gpu.ai-galaxy.cn/docs_v2/常见问题.html)、[智星云实例与保留磁盘费用说明](https://s.apifox.cn/b0fc397f-c455-4c9a-9d82-875fc48ae106/doc-6891582)。
