# 3D 线配准求 RT 的最基本公式

## 1. 先区分两个问题

如果你已经有了两组 3D 线：

```text
上一帧 3D 线集合  ->  当前帧 3D 线集合
```

那求 \(R,t\) 确实更像 ICP 点云配准。

只是 ICP 里配准的是点，这里配准的是线。

目标是找一个刚体变换：

\[
\mathbf{X}' = R\mathbf{X}+\mathbf{t}
\]

让第一组 3D 线经过 \(R,t\) 变换后，尽量和第二组 3D 线重合。

## 2. 一条 3D 线怎么表示

一条 3D 线可以用：

- 线上一点：\(\mathbf{p}\)
- 单位方向：\(\mathbf{d}\)

表示为：

\[
L=(\mathbf{p},\mathbf{d})
\]

其中：

\[
\|\mathbf{d}\|=1
\]

如果线由两个端点 \(\mathbf{p}_1,\mathbf{p}_2\) 给出，那么：

\[
\mathbf{d}=
\frac{\mathbf{p}_2-\mathbf{p}_1}
{\|\mathbf{p}_2-\mathbf{p}_1\|}
\]

## 3. RT 对 3D 线的作用

点的变换：

\[
\mathbf{p}'=R\mathbf{p}+\mathbf{t}
\]

方向的变换：

\[
\mathbf{d}'=R\mathbf{d}
\]

所以一条线经过 \(R,t\) 以后：

\[
L=(\mathbf{p},\mathbf{d})
\quad \rightarrow \quad
L'=(R\mathbf{p}+\mathbf{t}, R\mathbf{d})
\]

这就是 3D 线和 \(R,t\) 最基本的关系。

## 4. 线配准的目标

假设源线是：

\[
L_i^s=(\mathbf{p}_i^s,\mathbf{d}_i^s)
\]

目标线是：

\[
L_i^t=(\mathbf{p}_i^t,\mathbf{d}_i^t)
\]

希望经过 \(R,t\) 后：

\[
R\mathbf{d}_i^s \approx \mathbf{d}_i^t
\]

\[
R\mathbf{p}_i^s+\mathbf{t}
\quad \text{落在目标线 } L_i^t \text{ 上}
\]

## 5. 方向误差

两条线方向应该一致或反向一致。

常用方向误差：

\[
e_{\text{dir},i}
=
\left\|
\mathbf{d}_i^t \times R\mathbf{d}_i^s
\right\|
\]

如果两条方向平行，则叉乘为 0。

所以可以最小化：

\[
\min_R
\sum_i
\left\|
\mathbf{d}_i^t \times R\mathbf{d}_i^s
\right\|^2
\]

## 6. 点到线距离误差

点 \(\mathbf{x}\) 到目标线 \(L_i^t=(\mathbf{p}_i^t,\mathbf{d}_i^t)\) 的距离是：

\[
d(\mathbf{x},L_i^t)
=
\left\|
(\mathbf{I}-\mathbf{d}_i^t{\mathbf{d}_i^t}^T)
(\mathbf{x}-\mathbf{p}_i^t)
\right\|
\]

其中：

\[
\mathbf{I}-\mathbf{d}_i^t{\mathbf{d}_i^t}^T
\]

表示把误差投影到垂直于目标线方向的平面上。

源线上的点变换后是：

\[
\mathbf{x}_i = R\mathbf{p}_i^s+\mathbf{t}
\]

所以线配准的位置误差是：

\[
e_{\text{pos},i}
=
\left\|
(\mathbf{I}-\mathbf{d}_i^t{\mathbf{d}_i^t}^T)
(R\mathbf{p}_i^s+\mathbf{t}-\mathbf{p}_i^t)
\right\|
\]

## 7. 最基本的线配准目标函数

综合方向误差和位置误差：

\[
\min_{R,\mathbf{t}}
\sum_i
\left\|
\mathbf{d}_i^t \times R\mathbf{d}_i^s
\right\|^2
+
\lambda
\sum_i
\left\|
(\mathbf{I}-\mathbf{d}_i^t{\mathbf{d}_i^t}^T)
(R\mathbf{p}_i^s+\mathbf{t}-\mathbf{p}_i^t)
\right\|^2
\]

其中：

- 第一项：让线方向对齐；
- 第二项：让源线变换后靠近目标线；
- \(\lambda\)：控制位置误差权重。

这就是“线版 ICP”的核心公式。

## 8. 如果线有两个端点

如果源线有两个端点：

\[
\mathbf{p}_{1,i}^s,\mathbf{p}_{2,i}^s
\]

可以让两个端点都靠近目标线：

\[
\min_{R,\mathbf{t}}
\sum_i
\sum_{k=1}^{2}
\left\|
(\mathbf{I}-\mathbf{d}_i^t{\mathbf{d}_i^t}^T)
(R\mathbf{p}_{k,i}^s+\mathbf{t}-\mathbf{p}_i^t)
\right\|^2
\]

这比只用一个点更稳定。

## 9. 线 ICP 的流程

线 ICP 和点 ICP 类似：

```text
1. 给一个初始 R,t
2. 用 R,t 把源 3D 线变换过去
3. 给每条源线找最近的目标线
4. 用匹配好的线重新求 R,t
5. 重复 2-4，直到误差不再下降
```

找最近目标线时，一般同时看：

\[
\text{方向差} =
1-\left|
{\mathbf{d}_i^t}^T R\mathbf{d}_i^s
\right|
\]

和：

\[
\text{点到线距离}
=
d(R\mathbf{p}_i^s+\mathbf{t}, L_i^t)
\]

## 10. 和 3D-2D 求位姿的区别

如果你只有：

```text
3D 线 + 图像 2D 线
```

那是另一个问题，叫 3D-2D line pose，类似 PnL。

那时才需要相机内参 \(K\)，并使用：

\[
\mathbf{n}=K^T\boldsymbol{\ell}
\]

\[
\mathbf{n}^T(R\mathbf{P}_w+\mathbf{t})=0
\]

这里的 \(\boldsymbol{\ell}\) 是图像 2D 线。

但如果你说的是：

```text
3D 线集合 配准 3D 线集合
```

那就不需要 \(K\)，核心就是线到线配准。

## 11. 最简总结

3D 线配准求 RT，记住这几条就够了：

\[
\mathbf{p}'=R\mathbf{p}+\mathbf{t}
\]

\[
\mathbf{d}'=R\mathbf{d}
\]

\[
e_{\text{dir}}
=
\left\|
\mathbf{d}^t \times R\mathbf{d}^s
\right\|
\]

\[
e_{\text{pos}}
=
\left\|
(\mathbf{I}-\mathbf{d}^t{\mathbf{d}^t}^T)
(R\mathbf{p}^s+\mathbf{t}-\mathbf{p}^t)
\right\|
\]

整体意思是：

```text
像 ICP 点云配准一样，
只是点误差换成了线方向误差 + 点到线距离误差。
```
