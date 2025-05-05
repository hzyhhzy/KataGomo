# Latest release https://github.com/hzyhhzy/KataGomo/releases/tag/ConnectSix_20250505

# Training
Training scripts used are uploaded here:   
https://github.com/hzyhhzy/KataGomo/tree/ConnectSix2024/scripts   

## connectsix19x2
19x19 and smaller boards. ~5 million self-play games, with a computational cost of ~100 RTX 4090-days (approx. ¥3000 CNY).   
Transfer learning was applied from a Gomoku (Renju) model (b18c384nbt architecture).   
## connectsix25x
After completing connectsix19x2 training, short self-play training was on larger boards (up to 25x25).   


# Results
## 19x19
Black has a significant advantage. Opening library: `Connect6_19x19.MoveLimit113.Openings.7z`
Estimated minimum stones needed for a forced Black win: **109** (109 stones = 55 moves = 28 rounds).
![image](https://github.com/user-attachments/assets/21803cf1-a6eb-4e79-9440-2a7428a557fb)

## 18x18 or smaller
Highly likely to end in a draw
## 20x20
Insufficient computational analysis completed.
No significant advantage for Black currently observed.
## 21x21
Black has a stronger advantage.
Estimated stones needed for a forced Black win: **93** (93 stones = 47 moves = 24 rounds).
## 25x25
Black's advantage increases further.
Estimated stones needed for a forced Black win: **89** (89 stones = 45 moves = 23 rounds).