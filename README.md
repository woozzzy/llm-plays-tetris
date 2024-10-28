# llm-plays-tetris

## Inspiration
We drew inspiration from having a curiosity in making an AI play games. As Tetris fans, this project was an opportune moment to train an LLM as Tetris can be fully represented in text.

We also enjoyed watching Summoning Salt's 2-hour long video, [The History of Tetris World Records](https://www.youtube.com/watch?v=mOJlg8g8_yw) and thought this project was a fun extension.

## How we built it

Using the current #1 globally ranked Tetrio player, we used [rxtile's](https://ch.tetr.io/u/rtxile/blitz) Blitz mode gameplay to teach our LLM to "play like a pro." We then extracted each frame from the gameplay and identified the following parameters:

- the current state of the board
- the current Tetrimino
- the next Tetrimino
- the held Tetrimino

Because Tetris can be fully represented in text, we parsed the parameters as input for our pre-trained model to be fine-tuned. The model then learns how to infer the best placement of the Tetrimino. As each action is associated with a keystroke, the model outputs the keystrokes needed to move the Tetrimino. We then built out the script needed to communicate with the Raspberry Pi for it to input the keystrokes.

## Challenges we ran into

**Data Collection**: There are not any datasets on how to play Tetris "like a pro", so we had to manually learn how to extract the parameters we needed from a video. We had to ensure our data was as clean as possible in order for the model to be accurately fine-tuned.

**Efficiency and Latency**: Because we are streaming the video and grabbing thousands of frames, we ran into many latency problems. As a result, we had to ensure our code was as efficient as possible and did not rely on packages.

## Accomplishments that we're proud of

- We created our dataset using computer vision
- It works, despite the limited timeframe we had to program and the quantity of data we needed to obtain.
- First time using a Raspberry Pi
- It can also do T-Spins!

## What we learned

- Data quality matters most 
- Computer Vision data extraction techniques 
- Minimizing latency
- Translating keystrokes to Raspberry Pi

## What's next for LLM Plays Tetris

- Improving Performance issues
- Training on larger data
- Teaching the model how to play competitively (including speedrunning)
