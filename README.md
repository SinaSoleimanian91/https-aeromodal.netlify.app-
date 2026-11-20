
Airplane modal analysis app based on multibody dynamic model useful for post-balancing validation

An #airplane may require #balancing treatment during developmental flights or lifecycle. The balancing which often is done by adding counterweights changes natural frequencies. This change needs to be checked for preventing unwanted #vibrations, instability, flutter, etc. 
Here, I am sharing an app I developed recently, AeroModal, which can be run by Chrome, Firefox and Edge latest versions:
https://aeromodal.netlify.app

This app helps you to quickly validate whether the natural frequencies remain within the safety zone after balancing, or not.
First, you need to know whether the #unbalance source is on the left wing or on the right one.
Second, you need to find out the unbalance centre of mass (COM) and the amount of mass.
Third, you can do the balancing calculation by solving M₁ · (R_f + d₁) = M₂ · (R_f + d₂)
where M₁ and M₂ are either of added balancing mass or unbalance source, d₁ and d₂ are the distances of M1 and M2 from wing joints, and R_f is the fuselage radius.
Third, by correctly inputting the unbalance source and added mass characteristics, you can find out to what extent the natural frequencies have been changed.
#Python script is also attached.
