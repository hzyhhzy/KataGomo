# 2025.1 Create an independent repo for KataGomo
### Original repo for KataGomo: https://github.com/hzyhhzy/KataGomo_fork
### All release before 2025.1 https://github.com/hzyhhzy/KataGomo_fork/releases
### Original KataGo: https://github.com/lightvector/KataGo

# Introduction
KataGo is an AlphaZero-like AI for Go https://katagotraining.org/   
I modified KataGo to support various board games.  

With KataGo's high efficient reinforcement learning, it can reach the top-level AIs with very low cost (**probably <50 dollars**) for almost all board games. (Probably thousands of ELO stronger than other AIs if the game is not very popular)

Now this is probably the strongest AI for many games.    
Example: [Gomoku](https://github.com/hzyhhzy/KataGomo/releases/tag/Gomoku_20250206), [Hex](https://github.com/hzyhhzy/KataGomo/releases/tag/Hex_20250131),[Connect6](https://github.com/hzyhhzy/KataGomo/releases/tag/ConnectSix_20250505)...   
   
If you modified the rule of a board game or created a new board game, training KataGo is also an efficient way to learn how to play it and check whether it is balanced.   
Example: [Kill-all Go](https://github.com/hzyhhzy/KataGomo/tree/aliveWin),  [AntiGomoku (who first connect 5 loses)](https://github.com/hzyhhzy/KataGomo/tree/AntiGomoku)

Some mathematical problems can be represented as a 2-players or 1-player board game. KataGo may be also helpful.   
Example: [AngelProblem](https://github.com/hzyhhzy/KataGomo/tree/AngelProblem)

# How to train your own KataGo
## 1. Modify the engine's rule and compile.

### Train an existing game
If you want to train a game which is already in this repo (see the list below), just to use my source code is usually OK.

### Train a new game
If you want to train a new game, you should modify the rules in the c++ code. It is recommended to pick a branch (listed below) whose rule is the most similar to your game, and better to pick a branch not older than 2023.   
Most of the game logics are in `board.h/cpp` `boardhistory.h/cpp` `nninput.cpp`  ...   
You should understand the logic of `Board` and `BoardHistory` class.   
TODO: more detailed instruction of how to modify the engine.   

Usually the python(pytorch) code is not required to be modified. If the board of your game does not have 8 symmetries, you should modify `data_processing_pytorch.py`

### Compiling
Refer to https://github.com/lightvector/KataGo    . Both linux and windows are OK.   

## 2. Run training
For some of the games trained not earlier than 2024, the training scripts and configs used during the training are uploaded to `./scripts` of each branch.     
Some parameters should be changed before running: [Instruction](./training.md)


# Contact
Email: 2658628026@qq.com   
QQ: 2658628026 (reply fastest)   
Discord: hzy_sigmoid
Or simply in "issue"

# Games included (Updated 2025.5.12)
Only 2-player 2D board games without randomness   

Theoretically this algorithm is also suitable for games with randomness or 3D boards, but the modification of KataGo will be a massive work so I'm not planning to do it.     

Here is another example of reinforcement learning on a 1-player game **with randomness** [UmaAI: Umamusume AI](https://github.com/hzyhhzy/UmaAi)   

Source code of some games are hidden somewhere in my computer (marked as "lost") and not uploaded to Github (earlier than 2021)   
Some not popular games have not released the trained results. You can `issue` if you need it.

## 1. Popular games
| English | Chinese | Branch name| Latest major update | Latest release  | Notes       
|-------|-------|-------|-------|-------|-----------|
| Gomoku (Freestyle, Standard), Renju | 无禁手五子棋（六胜/六不胜）, 禁手五子棋 | [Gom2024](https://github.com/hzyhhzy/KataGomo/tree/Gom2024) | 2025.2| [Gomoku_20250206](https://github.com/hzyhhzy/KataGomo/releases/tag/Gomoku_20250206)|
| Animal Chess  |斗兽棋 |[AnimalChess2025](https://github.com/hzyhhzy/KataGomo/tree/AnimalChess2025)|2025.2|[Dandelion v2.4](https://github.com/lxsgx23/Dandelion-Chess/releases/tag/v2.4) |
| Tiaoqi (Chinese checkers)  |中国跳棋 |[tiaoqi](https://github.com/hzyhhzy/KataGomo/tree/tiaoqi)|2022.3|[20240406](https://github.com/hzyhhzy/KataGomo_fork/releases/tag/20240406) |
| Reversi  | 翻转棋 | [Reversi2023](https://github.com/hzyhhzy/KataGomo/tree/Reversi2023) | 2023.3| No release | Weaker than traditional engines
| Chinese Chess  | 中国象棋 | not opensourced | 2022 | No release | [PX0](https://github.com/official-pikafish/px0) is much better
| Chess  | 国际象棋 | lost | 2022 or earlier | No release | just a tiny test
  
## 2. Famous but less popular games
Variants of Go and Gomoku are not listed here but in the next section.
| English | Chinese | Branch name| Latest major update | Latest release  | Notes       
|-------|-------|-------|-------|-------|-----------|
| Hex | 六角棋/海克斯 | [Hex2024](https://github.com/hzyhhzy/KataGomo/tree/Hex2024) | 2025.1| [Hex_20250131](https://github.com/hzyhhzy/KataGomo/releases/tag/Hex_20250131)|
| Connect 6 | 六子棋/连六棋 | [ConnectSix2024](https://github.com/hzyhhzy/KataGomo/tree/ConnectSix2024) | 2024.12| [ConnectSix_20250505](https://github.com/hzyhhzy/KataGomo/releases/tag/ConnectSix_20250505)|
| Connect Four | （重力）四子棋 | [ConnectFour2024](https://github.com/hzyhhzy/KataGomo/tree/ConnectFour2024) | 2024.10| [ConnectFour_20241019](https://github.com/hzyhhzy/KataGomo_fork/releases/tag/ConnectFour_20241019)|
| Nogo | 不围棋 | [Nogo2025](https://github.com/hzyhhzy/KataGomo/tree/Nogo2025) | 2025.2| [Nogo20250219](https://github.com/hzyhhzy/KataGomo/releases/tag/Nogo20250219)|
| Ataxx | 同化棋 | [Ataxx2023](https://github.com/hzyhhzy/KataGomo/tree/Ataxx2023) | 2025.1| [Ataxx_20250131](https://github.com/hzyhhzy/KataGomo/releases/tag/Ataxx_20250131)|
| Amazons | 亚马逊棋 | [Amazons](https://github.com/hzyhhzy/KataGomo/tree/Amazons) | 2022 | [20240406](https://github.com/hzyhhzy/KataGomo_fork/releases/tag/20240406)|
| Breakthrough | 突破棋 | [breakthrough](https://github.com/hzyhhzy/KataGomo/tree/breakthrough) | 2022 | [20240406](https://github.com/hzyhhzy/KataGomo_fork/releases/tag/20240406)|
| Ultimate Tic-tac-toe | 终级井字棋 | [UltimateTictactoe2024](https://github.com/hzyhhzy/KataGomo/tree/UltimateTictactoe2024) | 2024.11 | [UltimateTictactoe_20241101](https://github.com/hzyhhzy/KataGomo/releases/tag/UltimateTictactoe_20241101)|
| ScoreFour/ConnectFour3D | 3D重力四子棋 | [ConnectFour3d](https://github.com/hzyhhzy/KataGomo/tree/ConnectFour3d) | 2024.11 | [ScoreFour_20250510](https://github.com/hzyhhzy/KataGomo/releases/tag/ScoreFour_20250510)| 
| DotsAndBoxes | 点格棋 | [DotsAndBoxes](https://github.com/hzyhhzy/KataGomo/tree/DotsAndBoxes) | 2024.3 | No release | Release soon
| Quoridor | 步步为营 | [Quoridor2024](https://github.com/hzyhhzy/KataGomo/tree/Quoridor2024) | 2024.10 | No release | Already exists similar AIs [Ka's AI](https://drive.google.com/drive/u/1/folders/10ZZLK9tDxJCG-0eV6wxiKF5g0RubCsxo)  
| Surakarta | - | [Surakarta](https://github.com/hzyhhzy/KataGomo/tree/Surakarta) | 2023.3 | No release | 
| Clobber | - | [Clobber2023](https://github.com/hzyhhzy/KataGomo/tree/Clobber2023) | 2023.3 | No release | 

## 3. Go variants
| English | Chinese | Branch name| Latest major update | Latest release  | Notes       
|-------|-------|-------|-------|-------|-----------|
| Capture Go | 吃子棋 | [CaptureGo2024](https://github.com/hzyhhzy/KataGomo/tree/CaptureGo2024) | 2024.10| [CaptureGo_20250509](https://github.com/hzyhhzy/KataGomo/releases/tag/CaptureGo_20250509)|
| Kill-all Go | 死活对局 | [LifeGo2024](https://github.com/hzyhhzy/KataGomo/tree/LifeGo2024) | 2024.10| [LifeGo_20241025](https://github.com/hzyhhzy/KataGomo_fork/releases/tag/LifeGo_20241025)|
| Hex Go | Hex棋盘围棋 | [HexGo2024](https://github.com/hzyhhzy/KataGomo/tree/HexGo2024) | 2024.10| No release |
| Not nearby last move | 禁止在上一手旁边 | [GoModify2a](https://github.com/hzyhhzy/KataGomo/tree/GoModify2a) | 2025.3| No release |
| Hex Capture Go | Hex棋盘吃子棋 | [HexCaptureGo](https://github.com/hzyhhzy/KataGomo/tree/HexCaptureGo) | 2022.1| No release |
| Weighted area Go | 目数加权围棋 | [weightGo](https://github.com/hzyhhzy/KataGomo/tree/weightGo) | 2022.1| No release |
| First capture my stone if no liberity | 没气先吃自己的棋子 | [FirstCaptureMe](https://github.com/hzyhhzy/KataGomo/tree/FirstCaptureMe) | 2022.1| No release |
| 1 capture = 4 score | 一子千金：吃1子4目 | [yiziqianjin](https://github.com/hzyhhzy/KataGomo/tree/yiziqianjin) | 2022.1| No release | 
| Who firstly can't move loses | 谁先没地方走谁输 | [yiziqianjin](https://github.com/hzyhhzy/KataGomo/tree/yiziqianjin) | 2022.1| No release | A special case of `yiziqianjin`: 1 capture = -1 score
| Cross opening： (odd,even) black, (even,odd) white | 交叉座子: 奇偶黑子，偶奇白子 | Lost | 2021 or earlier| No release |
|（No name #1）See the description| (无名魔改1) 见描述     |[GoModify1](https://github.com/hzyhhzy/KataGomo/tree/GoModify1) | 2024.10| No release |

## 4. Gomoku variants
Caro; Gomoku With Capture; Straight-Connect-Four; 墨棋; and many other modifications...

| English | Chinese | Branch name| Latest major update | Latest release  | Notes       
|-------|-------|-------|-------|-------|-----------|
| Caro | - | [Caro2024](https://github.com/hzyhhzy/KataGomo/tree/Caro2024)| 2025.2|[Gomoku_20250206](https://github.com/hzyhhzy/KataGomo/releases/tag/Gomoku_20250206)|
| Ban-location Gomoku | 禁点五子棋 | [GomBanLoc](https://github.com/hzyhhzy/KataGomo/tree/GomBanLoc)| 2022.3| No release |And will **NEVER** release
| Last move of five should be terminal | 连五的最后一手要在两端 | [GomNewRule1](https://github.com/hzyhhzy/KataGomo/tree/GomNewRule1) [GomNewRule2](https://github.com/hzyhhzy/KataGomo/tree/GomNewRule2)| 2022.3| No release
| Straight-Four | 直线四子棋 | lost | 2021.2| No release
| Different-line Gomoku | 双线五子棋  | [newgame_DifLineGomoku](https://github.com/hzyhhzy/KataGomo/tree/newgame_DifLineGomoku)| 2023.3| No release
| "Mo qi" | "墨棋" | [newgame_MoQiGomoku](https://github.com/hzyhhzy/KataGomo/tree/newgame_MoQiGomoku)| 2023.2| No release
| Five Count | 连五计数 | [fiveCount](https://github.com/hzyhhzy/KataGomo/tree/fiveCount)| 2022.3| No release
| Arithmetic progression Connect6 | 等差数列六子棋 | [EquDifGomoku](https://github.com/hzyhhzy/KataGomo/tree/EquDifGomoku)| 2022.2| No release
| Capture Gomoku | 吃子五子棋 | [capture_gomoku](https://github.com/hzyhhzy/KataGomo/tree/capture_gomoku)| 2021.8| No release
| Straight 5 and diagonal 6 / Straight 6 and diagonal 5 / Hex board Gomoku | 直线连五斜线连六 / 直线连六斜线连五 / Hex棋盘五子棋 | lost| 2021 or earlier | No release | Almost sure draw
| Forbid "4-3" Renju | 禁三四的禁手五子棋 | lost| 2021 or earlier | No release | Much more balanced than normal Renju which only not forbid "4-3"


## 5. Newly-invented games (except Go and Gomoku variants)
Some of these games are invented by Nijie(逆界) group (QQ group: 159281507)
| English | Chinese | Branch name| Latest major update | Latest release  | Notes       
|-------|-------|-------|-------|-------|-----------|
| AntiGomoku/AntiConnectN | 反五/四/三子棋 | [Caro2024](https://github.com/hzyhhzy/KataGomo/tree/Caro2024)| 2025.2|[Gomoku_20250206](https://github.com/hzyhhzy/KataGomo/releases/tag/Gomoku_20250206)|
| Zhen Qi / Quake Gomoku | 震棋 | [ZhenQi](https://github.com/hzyhhzy/KataGomo/tree/ZhenQi)| 2025.2|[Zhenqi20250218](https://github.com/hzyhhzy/KataGomo/releases/tag/Zhenqi20250218)|
| Xing Qi / Symmetry Game | 形棋 | [newgame_XingQi](https://github.com/hzyhhzy/KataGomo/tree/newgame_XingQi)| 2023.2|No release|
| King's move Connect 4 | 王步四子棋 | [con4type1](https://github.com/hzyhhzy/KataGomo/tree/con4type1)| 2022.3| No release
| WangMaLianXing / King-knight Gomoku | 王马连星 | lost| 2020.10|No release|
| Nijie's AntiGomoku | 逆界反五子棋 | [newgame_gomokuamazons](https://github.com/hzyhhzy/KataGomo/tree/newgame_gomokuamazons)| 2023.2| No release

## 6. Mathematical problems

| Description | Branch name| Latest major update | Results     
|-------|-------|-------|-------|     
|Min move for devil to win Angel Problem |[AngelProblem](https://github.com/hzyhhzy/KataGomo/tree/AngelProblem)|2022.4|  Min board for devil to win = 32x33. Min moves to win >100. Even on 101x101 board the fastest way is to force the angel to the wall|
| Analyzing the game in this article [Towards solving the 7-in-a-row game](https://arxiv.org/pdf/2107.05363) |[ProveSevenInARow](https://github.com/hzyhhzy/KataGomo/tree/ProveSevenInARow)|2024.5|The method in the article cannot finally prove it. More detailed: [Here](https://mathoverflow.net/questions/471302/can-gomokufive-in-a-row-draw-on-an-infinite-board-what-about-other-m-n-k-game)
| 2D "Dawson Chess": One stone should not be the neightbor of other stones |[DawsonChess](https://github.com/hzyhhzy/KataGomo/tree/DawsonChess)|2024.8| Chaotic
| Some kinds of special positions of Hex on infinite board. Detailed description and some results are here. [Mathoverflow](https://mathoverflow.net/questions/470376/connection-properties-of-a-single-stone-on-an-infinite-hex-board) |[HexTemplate](https://github.com/hzyhhzy/KataGomo/tree/HexTemplate)|2024.5| Still open. [Mathoverflow](https://mathoverflow.net/questions/470376/connection-properties-of-a-single-stone-on-an-infinite-hex-board) 
| If white can capture black's stones like Go, how many moves can black play before one stone being captured|[EscapeGo](https://github.com/hzyhhzy/KataGomo/tree/EscapeGo)|2022.2| About 25


# [Old README](./README_old.md) :More detailed introduction of some old branches