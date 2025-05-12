# Latest release: https://github.com/hzyhhzy/KataGomo/releases/tag/UltimateTictactoe_20241101  
# Rule
 
Brief introduction of the rule: https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe   
   
This AI includes a widely-accepted additional rule: If one should play in a finished (win, lose, or filled) sub-board, then he/she can play everywhere. Training katago without this additional rule is meaningless because it has been proved as first player win without this rule.   
   
This AI supports both with and without “Tiebreaker” rule.   
Tiebreaker: If no one connects 3 subboards, then who own more subboards wins.   


# Results
This game is complex enough to avoid being "weakly solved" by KataGo. 

The following images shows the winrates(up) and drawrates(down) of the first move  
### With Tiebreaker: First player 70%, Second player 14%, Draw 16%   
![image](https://github.com/user-attachments/assets/30632e76-5231-44bb-960c-2ee2c602eaff)
### Without Tiebreaker: First player 33%, Second player 11%, Draw 56%   
![image](https://github.com/user-attachments/assets/cc92a94c-b0b4-487e-9a06-032c8e816a1a)

# Training schedule
Scripts and parameters used in this run are in **./scripts**   
   
Total training cost: **~15 RTX4090\*day**       
    
One model for two rules (with and without "Tiebreaker"). Each rule has 50% selfplay games.    
Model size is b10c384nbt, ~12M selfplay games in total.   