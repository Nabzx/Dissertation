# Project Log

## Week 1 [w/c October]

- I watched a few videos on MDPs and neural networks from 3Blue1Brown etc. I think its a good idea to understand RL and the theory behind artificial intelligence before starting anything
- At this point I’m not coding much yet as im just trying to properly understand the theory before jumping into anything.
- My research question is basically confirmed 

## Week 2 [w/c November]
- Completed Interim Report
- Continued reading more RL and some game theory (like Axelrod). I think its pretty important to understand these concepts as they're quite relevant to RL. Especially understanding how cooperation actually emerges.
- As I've been focusing mainly on the theory behind single agent RL, I'm now looking at how that extends to MARL. It has been quite simple to understand so far. The maths is quite tricky however.
- The foundations of the project are implemented including the environment and agents.


## Week 3 [w/c December]
- I've had a look into MCP as a result of the last couple supervision meetings but I've come to the conclusion that it is not too relevant to my study
- Continuing to read papers on MARL
- Watched the Stanford CS234: Reinforcement Learning playlist in my free time

## Week 4 [w/c 06 January]
-  Continued reading papers, especially on communication and emergent behaviour
- Watched multiple youtube videos on reinforcement learning
- Ran further experiments with heuristic agents
- Started implementing testing pipeline

## Week 5 [w/c 20 January]
- Supervision session earlier in the week just covered keeping the project manageable and future objectives mainly, as well as my progress.
- Research into other frameworks (not pettingZoo) and maybe other alternatives to PPO
- Continued with testing pipeline - PPO agents are not fully functional yet, still some time to go before implementation

## Week 6 [w/c 03 February]
- Prep for interview next week.
- Supervision meeting (9th Feb) discussed project progress so far and things to mention in interview.
- Improvements made on metrics and measurements e.g. Jains Fairness
- PPO is being tested separately before integrating into main system
- Spent some more time consolidating knowledge on maths behind RL including policy gradients and advantage estimation

## Week 7 [w/c 17 February]
- Interview went ok, I got caught up in a few questions as mind went blank. Feedback was very good. Made me realise I should focus more on the actual implementation rather than the theory and maths.
- PPO still in testing
- At this point I still only have the base environment and heuristic agents but I have been playing around with reward shaping a bit.

## Week 8 [w/c 24 February]
- Supervision meeting where we discussed Interview and some advice to work on for the future. Also mentioned comparing trained vs untrained agents.
- Started integrating PPO into main environment properly
- Continued reading MARL papers (I am aware I mentioned I would focus less on theory but they are very interesting)


## Week 9 [w/c 03 March]
- Focusing mostly on making sure PPO works in main environment.
- Added a few more graphs for analysis
- Learnt about exploration and exploitation.

## Week 10 [w/c 17 March]
- Supervision meeting (9th March), talked about how to evaluate results more clearly and also mentioned 3D scaling.
- Added more graphs and analysis tracking
- Early testing is now running (100-1000 episodes)
- PPO ADDED
- Started writing dissertation - mainly on introduction and literature review so far.

## Week 11 [w/c 24 March]
- Supervision - mostly feedback on keeping research question clear
- Improved the environment by adding obstacles and partial observability
- Started implementing basic communication between agents
- Reward structure sort of added
- Looked into Mels.ai briefly but scaling to 3D in this project is highly unlikely.

## Week 12 [w/c 07 April]
- Improved PPO logic
- Built a proper experimental setup with both logging and tracking
- Started running 1000-5000 episode runs.
- Structure of files etc. is very messy but happy with my progress so far
- I started working on minigames for this project idea (like a sort of Squid Game style instead of Hunger Games)

## Week 14 [w/c 13 April - Cambridge Interview Prep]
- I just had my Cambridge interview and it mainly revolved around my final dissertation and research proposal (natural extension of this project) so I thought it was quite relevant to add here.
- I have spent an incredible amount of time revising for this interview - mainly brushing up on core RL concepts
- I went over PPO, actor critic and MARL challenges in the real world in detail
- I basically had to defend my project and also future proposal for 45 min (socratic method)
- This actually really helped clarify my understanding of my project and RL in general a lot.

## Week 14 [w/c 14 April]
- I've started running larger experiments from 1000 to 10000. 
- Also been running live demos of 5000 episodes
- added more detailed metrics of fairness, efficiency and cooperation.
- Added further scripts to analyse results after training
- Made the environment even more complex by adding 2 more agents and 45 obstacles in a now 25x25 grid (not 15x15 anymore). Also made it octagonal.
- Scrapped minigames as not essential to answering research q.
- Refactored my code to make a lot cleaner.

## Week 15 [w/c 21 April]
- Implemented communication between agents more properly (if anything I reduced it to make my research question more clear)
- Built a headless training pipeline to separate training from visualisation and also make results appear faster in the terminal (around a 2-3hr waittime instead of like 8 hours).
- Developed a much nicer live visualisation system with UI and graphs (live and working). I also tried to implement a playback and speed slider system but it was too complex and it was also unnecessary for this project.
- Added a comparison setup between pretrained and base agents.

## Week 15/16 [w/c 25 April]
- Very close to submission date but my project is essentially done. All that is left is to clean the code and add comments. I have finished my dissertation but just cleaning up the results section and adding citations correctly.





