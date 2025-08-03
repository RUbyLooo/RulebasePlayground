# atan和atan2的区别

`P = atan2(Y,X)` 返回 `Y` 和 `X` 的[四象限反正切](https://ww2.mathworks.cn/help/matlab/ref/atan2.html#buct8h0-4) (tan-1)，该值必须为实数。`atan2` 函数遵循当 `x` 在数学上为零（或者为 `0` 或 `-0`）时 `atan2(x,x)` 返回 `0` 的约定。

## 四象限反正切与二象限反正切

1. atan2(a,b)是4象限反正切，它的取值不仅取决于a/b的atan值，**还取决于点 (b, a) 落入哪个象限**

   1. 当点(b, a) 落入第一象限时，atan2(a,b)的范围是  0 ~ pi/2;
   2. 当点(b, a) 落入第二象限时，atan2(a,b)的范围是  pi/2 ~ pi;
   3. 当点(b, a) 落入第三象限时，atan2(a,b)的范围是  -pi/2～０;
   4. 当点(b, a) 落入第四象限时，atan2(a,b)的范围是 －pi～－pi/2

   

2. atan(a/b) 是2象限反正切，即a/b的atan值 

   1.  当 a/b > 0 时，atan(a/b)取值范围是 0 ~ pi/2；
   2.  当 a/b < 0 时，atan(a/b)取值范围是 -pi/2～０

