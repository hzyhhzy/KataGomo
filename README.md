# Latest release: https://github.com/hzyhhzy/KataGomo/releases/tag/UltimateTictactoe_20250616
# Rule
 
Brief introduction of the rule: https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe   
   
This AI includes a widely-accepted additional rule: If one should play in a finished (win, lose, or filled) sub-board, then he/she can play everywhere. Training katago without this additional rule is meaningless because it has been proved as first player win without this rule.   
   
This AI supports both with and without “Tiebreaker” rule.   
Tiebreaker: If no one connects 3 subboards, then who own more subboards wins.   


# Results:
The following images shows the winrates(up) and drawrates(down) of the first move
**With Tiebreaker: First player 78%, Second player 4%, Draw 18%**
![image](https://github.com/user-attachments/assets/d6ac335d-00fe-43e3-85cc-02ba0e0e3eb5)

**Without Tiebreaker: First player 21%, Second player 2%, Draw 77%**
![image](https://github.com/user-attachments/assets/16b72cff-0ccb-4be7-bcdd-c2febe28fbba)

# Training schedule
Scripts and parameters used in this run are in **./scripts/uttt2**   
   
Training cost of the latest run: **~10 RTX4090\*day**   
Total cost: **~25 RTX4090\*day**       
    
One model for two rules (with and without "Tiebreaker"). Each rule has 50% selfplay games.    
Model size is b10c384nbt, ~6M selfplay games for the latest run.   