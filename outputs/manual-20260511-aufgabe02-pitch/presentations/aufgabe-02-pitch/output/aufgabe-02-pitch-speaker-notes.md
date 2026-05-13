# Aufgabe 2 Pitch Speaker Notes

Target duration: 5-10 minutes. For 5 minutes, spend about 40 seconds per slide. For 10 minutes, spend 70-80 seconds on slides 2-6.

## Slide 1
Main line: "My main problem was not writing a robot script. It was getting trustworthy measurements, because every later decision depended on that."

Say:
- The assignment asks for Sim2Real comparison and a probabilistic endpoint model.
- The real work became: how do I know where the robot actually ended?
- I will present the problems, decisions, and consequences.

## Slide 2
Main line: "Odometry was useful, but not enough as an absolute final-position answer."

Say:
- I decided to use an overhead camera with three green markers.
- The tracker converts pixels to world meters using a homography.
- The pose estimator classifies the markers, computes x/y/yaw, and writes `results/latest_tracker_pose.csv`.
- This gave me a common physical coordinate frame for real final endpoints.

## Slide 3
Main line: "The next problem was repeatability: if the start changes, the endpoint statistics become contaminated."

Say:
- I added a start-pose gate before real runs.
- It requires 3 markers, a fresh pose, 1 second of stability, <= 0.04 m position error, and <= 4 deg yaw error.
- This made the run slower, but the data cleaner and easier to defend.

## Slide 4
Main line: "The same command did not mean the same runtime behavior in simulation and on hardware."

Say:
- In simulation, the script can stop when odometry reaches the target distance.
- On the real TurtleBot, the experiment executes timed or primitive commands and then measures with the tracker.
- I therefore logged both odometry and tracker fields, instead of pretending one source was perfect.

## Slide 5
Main line: "I chose a deliberately small empirical model because the task asked for a very rough prediction."

Say:
- I modeled measured motion primitives: forward commands and rotations.
- Example: F30 averaged about 0.302 m forward with small lateral error.
- The rotation primitives were systematic: a 90 deg command produced about 85 deg.
- The model is interpretable, but it does not include map uncertainty, trajectory uncertainty, or full x/y/yaw covariance.

## Slide 6
Main line: "The final route showed the limit of the model."

Say:
- The composed route used 10,000 Monte Carlo samples.
- Validation run 004 ended about 0.149 m away from the predicted mean and outside the 95% endpoint ellipse.
- This is not just a failure; it is the evidence the task wanted: open-loop endpoint prediction is brittle.

## Slide 7
Main line: "My conclusion is that the assignment made the need for SLAM visible."

Say:
- External measurement was necessary to answer the task.
- Start gating was necessary to make the repeated experiments fair.
- The empirical model was useful, but its failure on the route shows why a robot needs continuous localization feedback.
