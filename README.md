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

# Games included (Updated 2025.5.12)
Only 2-player 2D board games without randomness   

Theoretically this algorithm is also suitable for games with randomness or 3D boards, but the modification of KataGo will be a massive work so I'm not planning to do it.     

Here is another example of reinforcement learning on a 1-player game **with randomness** [UmaAI: Umamusume AI](https://github.com/hzyhhzy/UmaAi)

## 1. Popular games
| English | Chinese | Branch name| Latest major update | Latest release  | Notes       
|-------|-------|-------|-------|-------|-----------|
| Gomoku (Freestyle, Standard) | 五子棋（六胜/六不胜） | [Gom2024](https://github.com/hzyhhzy/KataGomo/tree/Gom2024) | 2025.2| [Gomoku_20250206](https://github.com/hzyhhzy/KataGomo/releases/tag/Gomoku_20250206)|
| Renju  | 连珠（有禁手五子棋） | [Gom2024](https://github.com/hzyhhzy/KataGomo/tree/Gom2024) | 2025.2| [Gomoku_20250206](https://github.com/hzyhhzy/KataGomo/releases/tag/Gomoku_20250206)|
| Animal Chess  |斗兽棋 |[AnimalChess2025](https://github.com/hzyhhzy/KataGomo/tree/AnimalChess2025)|2025.2|[Dandelion v2.4](https://github.com/lxsgx23/Dandelion-Chess/releases/tag/v2.4)
| Tiaoqi (Chinese checkers)  |中国跳棋 |[tiaoqi](https://github.com/hzyhhzy/KataGomo/tree/tiaoqi)|2022.3|[20240406](https://github.com/hzyhhzy/KataGomo_fork/releases/tag/20240406)
| Reversi  | 翻转棋 | [Reversi2023](https://github.com/hzyhhzy/KataGomo/tree/Reversi2023) | 2023.3| No release | Weaker than traditional engines
| Chinese Chess  | 中国象棋 | private | 2022 | No release | [PX0](https://github.com/official-pikafish/px0) is much better
  
## 2. Famous but less popular games

| English | Chinese | Branch name| Latest major update | Latest release  | Notes       
|-------|-------|-------|-------|-------|-----------|
| Hex | 六角棋/海克斯 | [Hex2024](https://github.com/hzyhhzy/KataGomo/tree/Hex2024) | 2025.1| [Hex_20250131](https://github.com/hzyhhzy/KataGomo/releases/tag/Hex_20250131)|
| Connect 6 | 六子棋/连六棋 | [ConnectSix2024](https://github.com/hzyhhzy/KataGomo/tree/ConnectSix2024) | 2024.12| [ConnectSix_20250505](https://github.com/hzyhhzy/KataGomo/releases/tag/ConnectSix_20250505)|
| Connect Four | （重力）四子棋 | [ConnectFour2024](https://github.com/hzyhhzy/KataGomo/tree/ConnectFour2024) | 2024.10| [ConnectFour_20241019](https://github.com/hzyhhzy/KataGomo_fork/releases/tag/ConnectFour_20241019)|
| Nogo | 不围棋 | [Nogo2025](https://github.com/hzyhhzy/KataGomo/tree/Nogo2025) | 2025.2| [Nogo20250219](https://github.com/hzyhhzy/KataGomo/releases/tag/Nogo20250219)|
| Ataxx | 同化棋 | [Ataxx2023](https://github.com/hzyhhzy/KataGomo/tree/Ataxx2023) | 2025.1| [Ataxx_20250131](https://github.com/hzyhhzy/KataGomo/releases/tag/Ataxx_20250131)|
| Amazons | 亚马逊棋 | [Amazons](https://github.com/hzyhhzy/KataGomo/tree/Amazons) | 2022 | [20240406](https://github.com/hzyhhzy/KataGomo_fork/releases/tag/20240406)|
| Breakthrough | - | [breakthrough](https://github.com/hzyhhzy/KataGomo/tree/breakthrough) | 2022 | [20240406](https://github.com/hzyhhzy/KataGomo_fork/releases/tag/20240406)|
| Ultimate Tic-tac-toe | 终级井字棋 | [UltimateTictactoe2024](https://github.com/hzyhhzy/KataGomo/tree/UltimateTictactoe2024) | 2024.11 | [UltimateTictactoe_20241019](https://github.com/hzyhhzy/KataGomo_fork/releases/tag/UltimateTictactoe_20241019)|
| TODO| 

# Contact
Email: 2658628026@qq.com   
QQ: 2658628026 (reply fastest)   
Discord: hzy_sigmoid



# Recent progresses (Latest update: 2024.10.26)

| Branch     | Introduction   | Stage        | Cost (RTX4090 \* Day) | Model size| Notes                  |
| :--------- | :------------- | :----------- | :--------------------- | :--------------------- |:--------------------- |
| [Gom2024](https://github.com/hzyhhzy/KataGo/tree/Gom2024)   | Gomoku and Renju       | **Paused** | 1500 | b28c512nbt  | Sponsored by Zhizi, will continue training months later  |
| [Hex2024](https://github.com/hzyhhzy/KataGo/tree/Hex2024)   | Hex game  | Finished   |  150 | b18c384nbt   | See the Release page  |
| [ConnectSix2024](https://github.com/hzyhhzy/KataGo/tree/ConnectSix2024)   | Towards solving (proving black win) 19x19 Connect Six  | **Training**   | 100 | b18c384nbt    | Will finish and release in ~ 5 days  |
| [Caro2024](https://github.com/hzyhhzy/KataGo/tree/Caro2024)   | Caro is a modified Gomoku  | Finished  | 60 | b10c384nbt     | https://github.com/hzyhhzy/KataGo/releases/tag/Caro_20240804 |
| [LifeGo2024](https://github.com/hzyhhzy/KataGo/tree/LifeGo2024)   | Life-and-death Go  | Finished   | 14  | b18c384nbt b28c512nbt   |   See the Release page |
| [CaptureGo2024](https://github.com/hzyhhzy/KataGo/tree/CaptureGo2024)   | Capture-Go  | Finished  | 18   | b18c384nbt b28c512nbt   | Will release in 2025.1 |
| [ConnectFour2024](https://github.com/hzyhhzy/KataGo/tree/ConnectFour2024)   | ConnectFour  | Finished  | 6  | b10c256nbt    | See the Release page |
| [HexGo2024](https://github.com/hzyhhzy/KataGo/tree/HexGo2024)   | Go on Hex board  | Finished   | 18 | b18c384nbt b28c512nbt    | Will release in ~ 5 days |
| [Quoridor2024](https://github.com/hzyhhzy/KataGo/tree/Quoridor2024)   | Quoridor  | Finished  | 7  | b10c128    | Someone has already trained an AI months ago, and found first player win  |
| [UltimateTictactoe2024](https://github.com/hzyhhzy/KataGo/tree/UltimateTictactoe2024)   | Ultimate Tic-tac-toe  | Finished   |  8 | b10c384nbt   | See the Release page |
| [DawsonChess](https://github.com/hzyhhzy/KataGo/tree/DawsonChess)   | 2D Dawson-chess and variants (Reverse Gomoku)  | Finished  | 1  | b10c384nbt    | Chaotic game |
| [GoModify1](https://github.com/hzyhhzy/KataGo/tree/GoModify1)   | A modification of Go  | Finished  | 6  | b18c384nbt    |    |
| [ProveSevenInARow](https://github.com/hzyhhzy/KataGo/tree/ProveSevenInARow)   | See the Readme in the branch  | Finished  |   | b10c256nbt    | https://mathoverflow.net/questions/471302/can-gomokufive-in-a-row-draw-on-an-infinite-board-what-about-other-m-n-k-game |
| [HexTemplate](https://github.com/hzyhhzy/KataGo/tree/HexTemplate)   | Some special openings for Hex | Finished  |    | b20c256   | https://mathoverflow.net/questions/470376/connection-properties-of-a-single-stone-on-an-infinite-hex-board |
| [DotsAndBoxes](https://github.com/hzyhhzy/KataGo/tree/DotsAndBoxes)  |  - | Finished  |  | b20c256     | Not released yet |



-------------------------------
# Brief introduction of recent branches (Latest update: 2024.8.21)
## Basic branches for fast modification to all kinds of games
### BWN2024
"Black-white No-score 2024"
For games which black and white plays alternately
### Movestone2024
For board games with multi-stage moves or irregular move order
For example: Chess(regard one move as two stages: choose and place), Amazons(3 stages), Ataxx(sometimes 1 stage, some times 2 stages), ConnectSix, ...

## Gomoku (Freestyle or Standard) and Renju
### Gom2024
Latest KataGo engine for Gomoku and Renju. Also called "KataGomo" and "KataGomoku".     
Recently a b28c512nbt net is training under the sponsorship of Zhizi(智子)(www.zhizigo.cn www.zhizigomoku.cn), which provides online paid KataGo services.   
The training started in 2024.5 and will finish in 2024.11. ~30\*RTX4090 for 10 hours every day. The strongest net is only available on Zhizi.      

### Gom2024NNUE
An experiment of combining NNUE and NN. (not very success)
Gomoku-NNUE: A fast and small network with special structure for gomoku.      
Gomoku-NNUE is firstly created by me (https://github.com/hzyhhzy/gomoku_nnue/tree/multiRules) and improved by dblue and used in Rapfi(https://github.com/dhbloo/rapfi)     
Input the result of a small MCTS (~300 nodes) of NNUE to the big katago net.   
Result:  The katago net become "as strong as NNUE+MCTS" in a very short time, however further progress is very slow.
If use transfer-training from a trained katago net, it become slightly stronger than the origin net with same nodes, but not worth its large CPU cost.

### GomNNUEnested
A branch to test MCTS of NNUE.
Regarding the result of a small MCTS (~300 nodes) of NNUE as a "meta-node", then apply MCTS to meta-nodes.
Katago NN is not used here. Only to use katago's MCTS


## Hex and related problems
https://en.wikipedia.org/wiki/Hex_(board_game)   

### Hex2024Maxmove
Latest KataHex code
Hex2024 with move limit settings, to estimate the length of the optimal game of Hex board on N\*N board   
Result: f(N) ~ 0.6 \* N^1.9   
https://mathoverflow.net/questions/302821/length-of-optimal-play-in-hex-as-a-function-of-size/477037#477037

### Hex2024
Latest release of Katago for Hex game (KataHex)
https://github.com/hzyhhzy/KataGo/releases/tag/Hex_20240812   

### HexTemplate
A special engine for "edge template". Special network is needed.   
This branch is to estimate the min width of 8~11th row "edge template" of Hex     
Result: f(n)>=9\*n-46 for n>=8, and probably f(n)=9\*n-46 for n<=11   
https://www.hexwiki.net/index.php/Open_problems_about_edge_templates   

### HexTemplateSolver
A special engine for single-stone escape problems (HexTemplate branch is more specific for "edge template"). Special network is needed.    
Some analysis and results: https://mathoverflow.net/questions/470376/connection-properties-of-a-single-stone-on-an-infinite-hex-board   


## Other Gomoku-like Games
Including Caro, ConnectSix, and maybe some math problems

### ConnectSix2024
KataGo for Connect Six
The 2022 ver has shown that black has great advantage on 21x21 or larger boards, but 19x19 was unknown.
Recently @eeoo has found that black has advantage on 19x19, so I started a new run in 2024.8.
I decide to strictly prove that black wins in Connect Six on 19x19 boards, collaborating with dblue using his ConnectSix Rapfi.

### Caro2024
Caro is a modified Gomoku rule, the only difference is that fives with two terminals blocked are not a win (OXXXXXO is not a win)   
Latest release: https://github.com/hzyhhzy/KataGo/releases/tag/Caro_20240804

### ProveSevenInARow
We all know black wins in 5-in-a-row(Gomoku) on an infinite board. And more than 5 in a row is very difficult which means easy to draw.   
Although it's very easy even for a beginner to draw in 6-in-a-row game, 7-in-a-row has not been proven to be a draw in infinite boards and remains to be an open problem for decades, showing a big gap between "theory" and "practice".   
https://arxiv.org/abs/2107.05363 This article shows a method to reduce the infinite board to 4\*n boards. If we can prove that there exist such one n, that white can draw on a 4\*n board (with some kinds of rules), then "7-in-a-row is a draw" is proved.  In this article the authors tried n<=14 but failed.    
This branch is to solve the game mentioned in this article on 4\*n boards with large n.     
However katago shows that the min n is around 30, which means it is almost impossible to strictly solve the 4\*n game. So this method is almost dead.    
More details was written here: https://mathoverflow.net/questions/471302/can-gomokufive-in-a-row-draw-on-an-infinite-board-what-about-other-m-n-k-game     

## Other games
### DotsAndBoxes
It's a game that black and white alternately play edges. It is hard to represent edges in Katago, so I uses a (2H-1)\*(2W-1) board to represent the Dots-and-boxes board
Result: This is a very chaotic game, no strategies, just brute-force search. (probably because it is almost an "impartical game"). Katago can learn almost nothing in the first 1/3 game and there is only some strategies near the end of the game

    
    
    
    
    
    
    
    
    
    
    
-------------------------
-------------------------
-------------------------
# Old

***2023.1.8 Engine based on Katago 1.12 start Modifying***   

| Tasks                                                  | Branch         | Stage        | Notes                              |
| :----------------------------------------------------- | :------------- | :----------- | :--------------------------------- |
| 1 Compile and run training                             | Kata2023       | Finished     | -                                  |
| 2 Chinese Rule Only(Remove JP rules)                   | -     | Skipped    |   |
| 3 CaptureGo(Remove Go rules)                           | -    | Skipped         |   |
| 4 Black-White board games(Remove Capture)              | BW2023/BWnoscore2023 | Finished(outdated)         | Using Gomoku as a representative, very easy to be modified to other games. "BWnoscore2023" branch also removed komi and score   |
| 4.0 Remove score and komi                              | BWnoscore2023 | Finished(outdated)         |    |
| 4.1 Gomoku                                             | Gom2023        | Todo         | New Katagomo engine                |
| 5 Chess-like games(Multi-stage moves)                  | Movestone2023  | Finished(outdated)         | Use Breakthrough as a representative, very easy to be modified to other games    |
| 5.0 Only X-axis symmetry. And flip Y-axis when playing white. (For chess-like games) | MovestoneXsym2023  | Finished    | subtreeBias and patternBonus are also removed |
 




***2022.1 Engine based on Katago TensorRT+GraphSearch***   

***Updated on 2022.4.9***   

Gomoku/Renju: "GomDevVCN" branch   
https://github.com/hzyhhzy/KataGo/tree/GomDevVCN      

TODO:   
先做改动小的，再做改动大的   

完全没改的分支：Go2022   
| 棋种                                                   | branch         | 状态         | 备注                               |
| :----------------------------------------------------- | :------------- | :----------- | :--------------------------------- |
| 1.各种变种围棋（保留中国规则，删掉日韩规则）           | CNrule2022     | -            | 基础分支                           |
| 1.1 加权点目                                           | weightGo       | finished     |
| 1.2 某个子不能死                                       |                | aborted      | 不做了，因为需要大量死活题作为开局 |
| 1.3.1 Hex 棋盘的围棋                                   | HexGo          | aborted      | 懒得做了                           |
| 1.3.2 Hex 棋盘的吃子棋                                 | HexCaptureGo   | finished     |
| 1.4 落子没气先提自己                                   | firstCaptureMe | finished     |
| 1.5 一子千金                                           | yiziqianjin    | finished     |
|                                                        |                |
| 2.需要提子但不需要点目的                               | CaptureGo2022  | -            | 基础分支                           |
| 2.1 吃子棋                                             | CaptureGo2022  | finished     |
| 2.2 反吃子棋（不围棋）                                 | CaptureGo2022  | finished     |
| 2.3 活一块就算赢(死活对局)                             | lifego2        | finished     |
| 2.4 黑棋活一块就赢，但是不能被吃子                     | aliveWin       | finished     |
| 2.5 吃子 n 子棋                                        |                | aborted      | 以前做过所以不做了                 |
| 2.6 白能吃黑，黑吃不掉白，黑棋最多多少手不被吃         | EscapeGo       | finished     |
| 2.7 谁先没地方下谁输                                   | yiziqianjin    | finished     | 视为特殊的一子千金                 |
|                                                        |
| 3.不需要提子，黑白交替落子的棋                         | Gom2022base    | -            | 基础分支，不跑训练                 |
| 3.0 不打算做的                                         |                |
| 3.0.2 重力四子棋                                       |                | aborted      | 以前做过，训练了很长时间           |
| 3.1 五子棋系列(各种规则)                               |                | -            |
| 3.1.0 主线(无禁,有禁,无禁六不胜)                       | GomDevVCN      | **training** | 100b256f，以后会跑分布式           |
| 3.1.2 禁点五子棋                                       | GomBanLoc      | finished     |
| 3.2 连五的个数                                         | fivecount      | finished     |
| 3.3 反 n 子棋                                          | AntiGomoku     | finished     |
| 3.4 Hex                                                | Hex2022        | finished     |
| 3.4.1 反 Hex                                           | Hex2022        | finished     |
| 3.5 等差数列 6 子棋                                    | EquDifGomoku   | finished     |
| 3.6 Angels and Devils game                             | AngelProblem   | finished     |
| 3.7 每一步必须在上一步的附近某些位置，满足某些条件获胜 |                | -            |
| 3.7.1 一种特殊四子棋                                   | con4type1      | finished     |
| 3.8 Reversi(奥赛罗，翻转棋)，反 Reversi                | Reversi2022    | finished     | 为了在botzone上打榜，还是做了      |
|                                                        |
| 4.一次走两步的棋，或者挪子的棋                         | Amazons        | -            | 把 Amazons 分支作为基础分支        |
| 4.0 国象，中象                                         |                | aborted      | 别人做过，且效果不好，所以不做     |
| 4.1 六子棋(Connect6)                                   | Connect6       | **training** |
| 4.2 中国跳棋                                           | tiaoqi         | finished     |
| 4.3 Amazons(亚马逊棋)                                  | Amazons        | finished     |
| 4.4 Breakthrough                                       | breakthrough   | finished     |
| 4.5 Ataxx(同化棋)                                      | Ataxx          | finished     |
| 4.6 各种极简象棋变种                                   |                | todo         |
|                                                        |
| 5.单人游戏                                             | -              |              |
| 5.1 消消乐                                             | xiaoxiaole     | todo         |
|                                                        |
| 6.完全信息，但是有随机性                               |                | todo         |




***Updated on 2021.4.5***
Now ***Gomoku/Renju*** code is uploading in "Gomoku" branch.




----------- 2020.7.26 ----------

Now there are gomoku(freestyle,standard,renju,caro), reversi, connect6, breakthrough, hex, "four in a row", Chinese checkers, and many board games few people play.

Now it's the strongest program on this games as I know: 
all kinds of gomoku(compared with Embryo)
connect6(compared with gzero)
hex(compared with gzero and)
Chinese checkers(compared with Shangxin Tiaoqi)

