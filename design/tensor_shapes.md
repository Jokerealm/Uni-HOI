# Tensor Shapes

默认配置：

- `B`: batch size
- `T = 8`
- `H = W = 256`
- `P = 16`
- `h = H / P = 16`
- `w = W / P = 16`
- `D = 512`
- `Nh = 1024`
- `No = 1024`
- `J = 22`
- `Cc = 4`

由此得到：

- 视频 token 长度：
  - `L_v = T * h * w = 8 * 16 * 16 = 2048`
- 状态 token 长度：
  - `L_s = Nh + No + T*J + T + T`
  - `= 1024 + 1024 + 176 + 8 + 8 = 2240`

## 1. 输入条件

| 变量 | Shape | 来源 | 去向 |
|---|---|---|---|
| `rgb` | `[B,T,3,H,W]` | Step1 cropped RGB | `condition_video` |
| `masks_human` | `[B,T,1,H,W]` | Step1 human mask | `condition_video`，`human_visible` |
| `masks_object` | `[B,T,1,H,W]` | Step1 object mask | `condition_video`，`silhouette/depth` loss |
| `depth` | `[B,T,1,H,W]` | Step1 aligned depth | `condition_video`，`object_depth` loss |
| `M_p` | `[B,T,1,H,W]` | Step1 primary region | `condition_video` |
| `M_s` | `[B,T,1,H,W]` | Step1 secondary region | `condition_video` |
| `M_object` | `[B,T,1,H,W]` | Step1 object region | `condition_video` |
| `keypoint_heatmaps` | `[B,T,1,H,W]` | 2D keypoints rasterization | `condition_video`，`joint_heat` loss |
| `condition_video` | `[B,T,10,H,W]` | 上述拼接 | `ConditionEncoder` |

## 2. Video Branch

| 变量 | Shape | 来源 | 去向 |
|---|---|---|---|
| `video_target` | `[B,T,6,H,W]` | `human_visible + teacher_object_video` | `VideoLatentCodec` |
| `video_target_tokens` | `[B,2048,512]` | `video_target` patchify + embed | FM target |
| `video_noise` | `[B,2048,512]` | Gaussian noise | FM interpolation |
| `video_xt` | `[B,2048,512]` | `t*x1 + (1-t)*noise` | model 输入 |
| `condition_tokens` | `[B,2048,512]` | `condition_video` 编码 | video cross conditioning |
| `geometry_tokens` | `[B,2048,512]` | `geometry_maps` 编码 | 3D -> 2D conditioning |
| `video_velocity` | `[B,2048,512]` | model 输出 | FM 主损失 |
| `decoded_video` | `[B,T,6,H,W]` | `x1_hat_v` 解码 | split 成 human/object branch |

## 3. State Branch

| 变量 | Shape | 来源 | 去向 |
|---|---|---|---|
| `G_h` | `[B,1024,14]` | teacher / bootstrap | `HOIStateCodec` |
| `G_o` | `[B,1024,14]` | teacher / bootstrap | `HOIStateCodec` |
| `joints_3d` | `[B,8,22,3]` | Step1 / SMPL | `HOIStateCodec` |
| `object_poses` | `[B,8,4,4]` | sequence pose fits | `HOIStateCodec` |
| `contact_signature` | `[B,8,4]` | hand-object distance/contact | `HOIStateCodec` |
| `state_target_tokens` | `[B,2240,512]` | state encode | FM target |
| `state_noise` | `[B,2240,512]` | Gaussian noise | FM interpolation |
| `state_xt` | `[B,2240,512]` | `t*x1 + (1-t)*noise` | model 输入 |
| `state_velocity` | `[B,2240,512]` | model 输出 | FM 主损失 |
| `decoded_state` | structured | `x1_hat_s` 解码 | render / geometry / state losses |

## 4. State Token 拆分

| token group | Shape |
|---|---|
| human Gaussian tokens | `[B,1024,512]` |
| object Gaussian tokens | `[B,1024,512]` |
| joint tokens | `[B,176,512]` |
| object motion tokens | `[B,8,512]` |
| contact tokens | `[B,8,512]` |

## 5. Geometry Projection

`GeometryProjector` 产出：

| 变量 | Shape | 含义 |
|---|---|---|
| `geometry_maps` | `[B,T,5,16,16]` | 当前 3D state 投影到 token 网格 |
| `joint_coords` | `[B,T,J,2]` | 3D joints 投影点 |
| `object_centers` | `[B,T,2]` | 物体投影中心 |

`geometry_maps` 五个通道：

- `ch0`: joint heat
- `ch1`: joint depth
- `ch2`: object silhouette
- `ch3`: object depth
- `ch4`: contact heat

## 6. 时空分离 Attention

视频分支不再对 `[B, 2048, D]` 直接做全局自注意力。

当前实现是：

1. reshape 到 `[B, T, h*w, D]`
2. 先做 spatial attention
   - 实际 attention shape：`[B*T, h*w, D]`
   - 即 `[B*8, 256, 512]`
3. 再做 temporal attention
   - 实际 attention shape：`[B*h*w, T, D]`
   - 即 `[B*256, 8, 512]`

这样显存和计算量都明显低于直接在 `2048` token 上做全 self-attention。
