*==============================================================================
* PEAD × Attention × LLM signal
* 数据：build/analysis_llm.dta（2004-2026）
*==============================================================================

version 18
clear all


cd "/Users/brittanyan/Desktop/PEADearning surprise部分"


*------------------------------------------------------------------------------
* 1. 读入与检查
*------------------------------------------------------------------------------
use "build/analysis_llm.dta", clear

describe                                  // 列名和类型
summarize                                 // 变量的基本统计
codebook att llm_first llm_avg, compact   // 取值范围和缺失

count                                     // 样本量
tab year                                  // 年份分布
tab ff10                                  // 行业分布
distinct permno                           // 多少家公司


*------------------------------------------------------------------------------
* 2. 变量说明
*------------------------------------------------------------------------------
* car_ann_o2o    earnings annoucement 窗口 [d, d+1] 的CAR，open-to-open
* car_drift_o2o  drift窗口 [d+2, d+61] 的累计异常收益
* sue_rank       盈余意外的当季十分位，缩放到 [0,1]，系数读作 D10 减 D1
* att            11 − NRANK，1–10，数值越大注意力越充裕
* llm_first      事件窗口内时间戳最早那条新闻的 LLM 预测
* llm_avg        事件窗口内所有新闻的 LLM 预测按条平均
* llm_n          事件窗口内的新闻条数
* date_id        公告日编号，聚类维度


*------------------------------------------------------------------------------
* 3. 控制变量global
*------------------------------------------------------------------------------
global controls size_dec bm_dec lnanalyst lag lag2 lag3 io evol epersist turn

display "$controls"                       // 检查global有没有定义成功


*------------------------------------------------------------------------------
* 4. 描述统计与相关性
*------------------------------------------------------------------------------
summarize car_ann_o2o car_drift_o2o sue_rank att llm_first llm_avg llm_n, detail

pwcorr sue_rank att llm_first llm_avg car_ann_o2o car_drift_o2o, sig star(0.05)

* LLM signal 在 SUE 十分位上的形态
table sue_dec, statistic(mean llm_first) statistic(mean llm_avg) statistic(freq)

* 注意力十分位上的漂移
table att, statistic(mean car_drift_o2o) statistic(freq)


*------------------------------------------------------------------------------
* 5. 主回归：llm_first
*   三档设定 × 两个窗口
*------------------------------------------------------------------------------
eststo clear

* --- 公告窗口 [d, d+1] ---

* (1) 最简
reghdfe car_ann_o2o sue_rank att llm_first c.att#c.llm_first ///
    $controls, absorb(year month dow ff10) vce(cluster date_id)
eststo a1_first

* (2) 三个两两交互全齐
reghdfe car_ann_o2o sue_rank att llm_first ///
    c.att#c.llm_first c.sue_rank#c.att c.sue_rank#c.llm_first ///
    $controls, absorb(year month dow ff10) vce(cluster date_id)
eststo a2_first

* (3) 完整version：控制变量分别与 SUE、LLM 交互
reghdfe car_ann_o2o sue_rank att llm_first ///
    c.att#c.llm_first c.sue_rank#c.att c.sue_rank#c.llm_first ///
    $controls c.($controls)#c.sue_rank c.($controls)#c.llm_first, ///
    absorb(year month dow ff10) vce(cluster date_id)
eststo a3_first

* --- 漂移窗口 [d+2, d+61] ---

reghdfe car_drift_o2o sue_rank att llm_first c.att#c.llm_first ///
    $controls, absorb(year month dow ff10) vce(cluster date_id)
eststo d1_first

reghdfe car_drift_o2o sue_rank att llm_first ///
    c.att#c.llm_first c.sue_rank#c.att c.sue_rank#c.llm_first ///
    $controls, absorb(year month dow ff10) vce(cluster date_id)
eststo d2_first

reghdfe car_drift_o2o sue_rank att llm_first ///
    c.att#c.llm_first c.sue_rank#c.att c.sue_rank#c.llm_first ///
    $controls c.($controls)#c.sue_rank c.($controls)#c.llm_first, ///
    absorb(year month dow ff10) vce(cluster date_id)
eststo d3_first


*------------------------------------------------------------------------------
* 6. 主回归：llm_avg
*------------------------------------------------------------------------------

reghdfe car_ann_o2o sue_rank att llm_avg c.att#c.llm_avg ///
    $controls, absorb(year month dow ff10) vce(cluster date_id)
eststo a1_avg

reghdfe car_ann_o2o sue_rank att llm_avg ///
    c.att#c.llm_avg c.sue_rank#c.att c.sue_rank#c.llm_avg ///
    $controls, absorb(year month dow ff10) vce(cluster date_id)
eststo a2_avg

reghdfe car_ann_o2o sue_rank att llm_avg ///
    c.att#c.llm_avg c.sue_rank#c.att c.sue_rank#c.llm_avg ///
    $controls c.($controls)#c.sue_rank c.($controls)#c.llm_avg, ///
    absorb(year month dow ff10) vce(cluster date_id)
eststo a3_avg

reghdfe car_drift_o2o sue_rank att llm_avg c.att#c.llm_avg ///
    $controls, absorb(year month dow ff10) vce(cluster date_id)
eststo d1_avg

reghdfe car_drift_o2o sue_rank att llm_avg ///
    c.att#c.llm_avg c.sue_rank#c.att c.sue_rank#c.llm_avg ///
    $controls, absorb(year month dow ff10) vce(cluster date_id)
eststo d2_avg

reghdfe car_drift_o2o sue_rank att llm_avg ///
    c.att#c.llm_avg c.sue_rank#c.att c.sue_rank#c.llm_avg ///
    $controls c.($controls)#c.sue_rank c.($controls)#c.llm_avg, ///
    absorb(year month dow ff10) vce(cluster date_id)
eststo d3_avg


*------------------------------------------------------------------------------
* 7. 边际效应：LLM 的作用如何随注意力变化
*------------------------------------------------------------------------------
reghdfe car_drift_o2o sue_rank att llm_first ///
    c.att#c.llm_first c.sue_rank#c.att c.sue_rank#c.llm_first ///
    $controls, absorb(year month dow ff10) vce(cluster date_id)

margins, dydx(llm_first) at(att = (1(1)10))

marginsplot, yline(0) ///
    title("Marginal effect of LLM signal on drift, by attention") ///
    xtitle("ATT (higher = more attention available)") ///
    ytitle("dCAR[2,61] / dLLM")

graph export "build/margins_att_llm.png", replace width(1400)



*------------------------------------------------------------------------------
* 导出：简表 + 全表
*------------------------------------------------------------------------------

* 简表
esttab a1_first a2_first a3_first d1_first d2_first d3_first ///
    using "build/reg_llm_first.rtf", replace ///
    b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    keep(sue_rank att llm_first c.att#c.llm_first ///
         c.sue_rank#c.att c.sue_rank#c.llm_first) ///
    order(sue_rank att llm_first c.att#c.llm_first ///
          c.sue_rank#c.att c.sue_rank#c.llm_first) ///
    mtitles("ANN (1)" "ANN (2)" "ANN (3)" "DRIFT (1)" "DRIFT (2)" "DRIFT (3)") ///
    stats(N r2_within, fmt(%9.0gc %9.4f) labels("Observations" "Within R2")) ///
    title("Table 1. ATT x LLM, llm_first, O2O returns")



* 全表：所有系数
esttab a1_first a2_first a3_first d1_first d2_first d3_first ///
    using "build/reg_llm_first_full.rtf", replace ///
    b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) drop(o.*, relax) ///
    mtitles("ANN (1)" "ANN (2)" "ANN (3)" "DRIFT (1)" "DRIFT (2)" "DRIFT (3)") ///
    stats(N r2_within, fmt(%9.0gc %9.4f) labels("Observations" "Within R2")) ///
    title("Table A1. ATT x LLM, llm_first, all coefficients")

	
	
*------------------------------------------------------------------------------
* avg LLM signal
* 简表
esttab a1_avg a2_avg a3_avg d1_avg d2_avg d3_avg ///
    using "build/reg_llm_avg.rtf", replace ///
    b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    keep(sue_rank att llm_avg c.att#c.llm_avg ///
         c.sue_rank#c.att c.sue_rank#c.llm_avg) ///
    order(sue_rank att llm_avg c.att#c.llm_avg ///
          c.sue_rank#c.att c.sue_rank#c.llm_avg) ///
    mtitles("ANN (1)" "ANN (2)" "ANN (3)" "DRIFT (1)" "DRIFT (2)" "DRIFT (3)") ///
    stats(N r2_within, fmt(%9.0gc %9.4f) labels("Observations" "Within R2")) ///
    title("Table 2. ATT x LLM, llm_avg, O2O returns")

* 全表
esttab a1_avg a2_avg a3_avg d1_avg d2_avg d3_avg ///
    using "build/reg_llm_avg_full.rtf", replace ///
    b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) drop(o.*, relax) ///
    mtitles("ANN (1)" "ANN (2)" "ANN (3)" "DRIFT (1)" "DRIFT (2)" "DRIFT (3)") ///
    stats(N r2_within, fmt(%9.0gc %9.4f) labels("Observations" "Within R2")) ///
    title("Table A2. ATT x LLM, llm_avg, all coefficients")

