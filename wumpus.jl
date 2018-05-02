# SIMPLIFIED Wumpus environment:
# Envitonment: gold, two pits and wumpus
#	no agent orientation, no arrow, 

# Actions:move N, S, E, W, Grab

# Observations: stench, breeze, glitter

using POMDPs
using Distributions
using POMDPToolbox
using SARSOP

# Used to print the state like a grid
import Base.show

# Define if we are using the complete state space or a sample version
# Complete version: I never managed to complete the policy generation, it takes too much time
# Reduced version: Pits, Wumpus and Gold location are static. 
#   1536 states(pits, wumpus, gold static) takes roughly 25 seconds to run on my machine
#	24576 states(pits, wumpus static), It takes roughly 90 minutes to run on my machine


# SETTINGS
complete_space = true
fixed_pits = true

checks_enabled = false

load_problem = false
n_runs = 10
sarsop_simulation = false
normal_simulation = false
show_normal_simulation = false
generate_graph = true

struct WumpusState
    # Since we have two pits and we are not sure of their location, we can't use xP1,yP1 and xP2,yP2
    # If we represent the pits as 'is there a pit in x,y' The Observation/transition model become easier
    p11::Bool
    p12::Bool
    p13::Bool
    p14::Bool
    p21::Bool
    p22::Bool
    p23::Bool
    p24::Bool
    p31::Bool
    p32::Bool
    p33::Bool
    p34::Bool
    p41::Bool
    p42::Bool
    p43::Bool
    p44::Bool
    xA::Int64   # Agent position
    yA::Int64
    xW::Int64   # Wumpus position
    yW::Int64
    xGold::Int64    # Gold position
    yGold::Int64
    grabbed::Bool
end

# Function to print the state in a readable form
function Base.show(io::IO, s::WumpusState)
    # Draw the agent
    a = "A"
    
    # Draw the wumpus
    w = "W"
    
    border = "+---+---+---+---+\n"
    str = border
    for y = 4:-1:1
        row1 = "|"
        row2 = "|"
        for row = 1:2
            for x = 1:4
                if row == 1
                    if s.xA == x && s.yA == y
                        row1 = row1 * a
                    else
                        row1 = row1 * " "
                    end
                    if isPit(s,x,y)
                        row1 = row1 * "P"
                    else
                        row1 = row1 * " "
                    end
                    row1 = row1 * " |"
                else
                    if s.xW == x && s.yW == y
                        row2 = row2 * w
                    else
                        row2 = row2 * " "
                    end
                    if s.xGold == x && s.yGold == y
			if s.grabbed
                            row2 = row2 * "g"
			else
			    row2 = row2 * "G"
			end
                    else
                        row2 = row2 * " "
                    end
                    row2 = row2 * " |"
                end
            end
        end
        str = str  * row1 * "\n"
        str = str  * row2 * "\n"
        str = str  * border
    end
    print(io, str)
end

# Templates for creating states
    # Create two identical states, except for the result of a move action (move changes the agent position)
function WumpusStateMove(s::WumpusState, xA::Int64, yA::Int64)
    return WumpusState(s.p11, s.p12, s.p13, s.p14, s.p21, s.p22, s.p23, s.p24, s.p31, s.p32, s.p33, s.p34, s.p41, s.p42, s.p43, s.p44, xA, yA, s.xW, s.yW, s.xGold, s.yGold, s.grabbed)
end

    # Create two identical states, except for the result of a grab action (we set the completed flag)
function WumpusStateGrab(s::WumpusState, grabbed::Bool)
    return WumpusState(s.p11, s.p12, s.p13, s.p14, s.p21, s.p22, s.p23, s.p24, s.p31, s.p32, s.p33, s.p34, s.p41, s.p42, s.p43, s.p44, s.xA, s.yA, s.xW, s.yW, s.xGold, s.yGold, grabbed)
end



# Check if there is a pit (used in order to not to deal with s.pxy each time)
function isPit(s::WumpusState, x::Int64, y::Int64)
    if x == 1 && y == 1
        return s.p11
    elseif x == 1 && y == 2
        return s.p12
    elseif x == 1 && y == 3
        return s.p13
    elseif x == 1 && y == 4
        return s.p14
    elseif x == 2 && y == 1
        return s.p21
    elseif x == 2 && y == 2
        return s.p22
    elseif x == 2 && y == 3
        return s.p23
    elseif x == 2 && y == 4
        return s.p24
    elseif x == 3 && y == 1
        return s.p31
    elseif x == 3 && y == 2
        return s.p32
    elseif x == 3 && y == 3
        return s.p33
    elseif x == 3 && y == 4
        return s.p34
    elseif x == 4 && y == 1
        return s.p41
    elseif x == 4 && y == 2
        return s.p42
    elseif x == 4 && y == 3
        return s.p43
    elseif x == 4 && y == 4
        return s.p44
    else
        error("Location ($x,$y) not present in this world representation")
    end
end

# Determine if a square has breeze
function isBreeze(x::Int64, y::Int64, s::WumpusState)
    if (x == 1 && y  == 1)
        return s.p12 || s.p21
    elseif (x == 1 && y == 2)
        return s.p11 || s.p13 || s.p22
    elseif (x == 1 && y == 3)
        return s.p12 || s.p14 || s.p23
    elseif (x == 1 && y == 4)
        return s.p13 || s.p24
    elseif (x == 2 && y == 1)
        return s.p11 || s.p31 || s.p22
    elseif (x == 2 && y == 2)
        return s.p12 || s.p32 || s.p23 || s.p21
    elseif (x == 2 && y == 3)
        return s.p13 || s.p33 || s.p24 || s.p22
    elseif (x == 2 && y == 4)
        return s.p14 || s.p34 || s.p23
    elseif (x == 3 && y == 1)
        return s.p21 || s.p41 || s.p32
    elseif (x == 3 && y == 2)
        return s.p22 || s.p42 || s.p33 || s.p31
    elseif (x == 3 && y == 3)
        return s.p23 || s.p43 || s.p34 || s.p32
    elseif (x == 3 && y == 4)
        return s.p24 || s.p44 || s.p33
    elseif (x == 4 && y  == 1)
        return s.p31 || s.p42
    elseif (x == 4 && y == 2)
        return s.p41 || s.p43 || s.p32
    elseif (x == 4 && y == 3)
        return s.p42 || s.p44 || s.p33
    elseif (x == 4 && y == 4)
        return s.p43 || s.p34
    else
        return false
    end
end

function isStateBreeze(s::WumpusState)
    # Indicates if a certain state can emit the breeze observation
    if isBreeze(s.xA,s.yA,s)
        return true
    else
        return false
    end
end

function isStench(x::Int64, y::Int64, s::WumpusState)
    if (abs(x - s.xW) + abs(y - s.yW)) == 1
        return true
    end
    return false
end

function isStateStench(s::WumpusState)
    # Indicates if a certain state can emit the stench observation
    for x = 1:4
        for y = 1:4
            if (s.xA == x && s.yA == y) && isStench(x,y,s)
                return true
            end
        end
    end
    return false
end


function isStateGlitter(s::WumpusState)
    if (s.xGold == s.xA) && (s.yGold == s.yA)
        return true
    end
    return false
end


# Check that at most 2 pits are present

function checkTwoPits(v::Vector{Bool})
    count = 0
    for x in v
        if x
            count += 1
        end
    end
    if count == 2
        return true
    else
        return false
    end
end


### Model definition
type WumpusPOMDP <: POMDP{WumpusState, Symbol, Symbol} # POMDP{State, Action, Observation} all parametarized by Int64s
    r_action::Float64 # reward for making any action (default -1)
    r_inPit::Float64 # reward for the agent being in a pit (default -1000)
    r_inWumpusMouth::Float64 # reward for bein eaten (default -1000)
    r_grabGold::Float64 # reward for taking the gold (default 1000)
    discount_factor::Float64 # discount
end

# Default model constructor
function WumpusPOMDP()
    return WumpusPOMDP(-1, -1000, -1000, 1000, 0.90)
end

#wumpus = WumpusPOMDP()

function POMDPs.discount(pomdp::WumpusPOMDP)
    return pomdp.discount_factor
end


####
# STATE SPACE DEFINITION:
# POMDPs.states(::MyPOMDP)
#   Is the state container. This function is used in order to create all the states present in the problem and will serve as a 
#   future container for all the states (If we want to have an array of all the possible states, this is the function to call)
# POMDPs.state_index(::MyPOMDP, ::State)
#   Is the indexing function. Each state must return a different index. The domain of the indexing funcion must be equal to the size 
#   of the state space
# POMDPs.n_states(::MyPOMDP)
#   Used (I think from SARSOP) in order to retrieve the number of states of the problem
# POMDPs.initial_state_distribution(::MyPOMDP)
#   Used by the SARSOP solver as an initial belief state distribution. It will indicate which are the entry point of the state machine that will 
#   be generated
# POMDPs.isterminal(::MyPOMDP, ::State)
#   Used by the SARSOP solver in order to determine whether the state is a terminal one
####
if complete_space
    # State generation for the complete state space
    function generateStates()
        counter = 0
        s = WumpusState[]   # Create an empty array of states
        for p11 in [true,false], p12 in [true,false], p13 in [true,false], p14 in [true,false], p21 in [true,false], p22 in [true,false], p23 in [true,false], p24 in [true,false], p31 in [true,false], p32 in [true,false], p33 in [true,false], p34 in [true,false], p41 in [true,false], p42 in [true,false], p43 in [true,false], p44 in [true,false]
	    if checkTwoPits([p11,p12,p13,p14,p21,p22,p23,p24,p31,p32,p33,p34,p41,p42,p43,p44])
	        for xA=1:4, yA=1:4, xW=1:4, yW=1:4, xGold=1:4, yGold=1:4, grabbed in [true,false]
                tmpState = WumpusState(p11,p12,p13,p14,p21,p22,p23,p24,p31,p32,p33,p34,p41,p42,p43,p44,xA,yA,xW,yW,xGold,yGold,grabbed)
	            push!(s, tmpState)
	            counter += 1
	        end
	    end
        end
        println("$counter states defined")
        return s
    end
    function calculateIndex(pomdp::WumpusPOMDP, s::WumpusState)
        pitarray = [s.p11, s.p12, s.p13, s.p14, s.p21, s.p22, s.p23, s.p24, s.p31, s.p32, s.p33, s.p34, s.p41, s.p42, s.p43, s.p44]
        firstpit = 0
        for i = 1:size(pitarray,1)
	    if pitarray[i]
	        firstpit = i
	        break
	    end
        end
           
        secondpit = 0
        for i = firstpit+1:size(pitarray,1)
	    if pitarray[i]
	        secondpit = i
	        break
	    end
        end
        
        pitIndex = secondpit - firstpit
        counter = 15
        for i = (firstpit-1):-1:1
	    pitIndex += counter
	    counter -= 1
        end
        
        return sub2ind((120,4,4,4,4,4,4,2), pitIndex, s.xA, s.yA, s.xW, s.yW, s.xGold, s.yGold, Int(s.grabbed+1))
    end
else
    # State generation for the reduced state spaces
    function generateStates()
        counter = 0
        s = WumpusState[]
        if fixed_pits
	        # Fixed pits:
            p11 = false
            p12 = false
            p13 = false
            p14 = false
            p21 = false
            p22 = false
            p23 = true
            p24 = false
            p31 = false
            p32 = true
            p33 = false
            p34 = false
            p41 = false
            p42 = false
            p43 = false
            p44 = false
            if checkTwoPits([p11,p12,p13,p14,p21,p22,p23,p24,p31,p32,p33,p34,p41,p42,p43,p44])
                for xA=1:4, yA=1:4, xW=1:4, yW=2, xGold=1:4, yGold=1:4, grabbed in [true,false]
                    tmpState = WumpusState(p11,p12,p13,p14,p21,p22,p23,p24,p31,p32,p33,p34,p41,p42,p43,p44,xA,yA,xW,yW,xGold,yGold,grabbed)
                    push!(s, tmpState)
                    counter += 1
                end
            else
                println("Check pits combination for the state generator!")
            end
        else
            # Variable pits
            for p11 in [true,false], p12 in [true,false], p13 in [true,false], p14 in [true,false], p21 in [true,false], p22 in [true,false], p23 in [true,false], p24 in [true,false], p31 in [true,false], p32 in [true,false], p33 in [true,false], p34 in [true,false], p41 in [true,false], p42 in [true,false], p43 in [true,false], p44 in [true,false]
            	if checkTwoPits([p11,p12,p13,p14,p21,p22,p23,p24,p31,p32,p33,p34,p41,p42,p43,p44])
                    for xA=1:4, yA=1:4, xW=1, yW=1:4, xGold=1:4, yGold=3, grabbed in [true,false]
                        tmpState = WumpusState(p11,p12,p13,p14,p21,p22,p23,p24,p31,p32,p33,p34,p41,p42,p43,p44,xA,yA,xW,yW,xGold,yGold,grabbed)
	                    push!(s, tmpState)
	                    counter += 1
	                end
	            end
	        end
        end
        
        println("$counter states defined")
        return s
    end
    
    function calculateIndex(pomdp::WumpusPOMDP, s::WumpusState)
        pitarray = [s.p11, s.p12, s.p13, s.p14, s.p21, s.p22, s.p23, s.p24, s.p31, s.p32, s.p33, s.p34, s.p41, s.p42, s.p43, s.p44]
        firstpit = 0
        for i = 1:size(pitarray,1)
	    if pitarray[i]
	        firstpit = i
	        break
	    end
        end
           
        secondpit = 0
        for i = firstpit+1:size(pitarray,1)
	    if pitarray[i]
	        secondpit = i
	        break
	    end
        end
        
        pitIndex = secondpit - firstpit
        counter = 15
        for i = (firstpit-1):-1:1
	    pitIndex += counter
	    counter -= 1
        end
        if fixed_pits
            return sub2ind((1,4,4,4,1,4,4,2), 1, s.xA, s.yA, s.xW, 1, s.xGold, s.yGold, Int(s.grabbed+1))
        else
            return sub2ind((120,4,4,1,4,4,1,2), pitIndex, s.xA, s.yA, 1, s.yW, s.xGold, 1, Int(s.grabbed+1))
        end
    end

end

allStates = generateStates()

function POMDPs.states(pomdp::WumpusPOMDP)
    return allStates
end

function POMDPs.state_index(pomdp::WumpusPOMDP, s::WumpusState)
    return calculateIndex(pomdp, s)
end

POMDPs.n_states(pomdp::WumpusPOMDP) = length(allStates)


function POMDPs.initial_state_distribution(pomdp::WumpusPOMDP)
    allStates = []
    counter = 0
    for s in POMDPs.states(pomdp)
        if s.xA ==1 && s.yA == 1 && s.grabbed == false
            push!(allStates, s)
            counter += 1
        end
    end
    return SparseCat(allStates, zeros(counter) + 1/counter)
end

function POMDPs.isterminal(pomdp::WumpusPOMDP, s::WumpusState)
   
    if isPit(s, s.xA, s.yA)
        # I am dead
        return true
    elseif (s.xW == s.xA) && (s.yW == s.yA)
        # I am dead
        return true
    elseif s.grabbed
        # I grabbed the gold
        return true
    else
        return false
    end
        
end

####
# ACTION SPACE DEFINITION:
# All the used function are used in the same way as explained for state space definition
# POMDPs.actions(::MyPOMDP)
# POMDPs.action_index(::MyPOMDP, ::Action)
# POMDPs.n_actions(::MyPOMDP)
####

POMDPs.actions(pomdp::WumpusPOMDP) = [:N, :S, :E, :W, :grab, :noop]

function POMDPs.action_index(::WumpusPOMDP, a::Symbol)
    if a == :N
        return 1
    elseif a == :E
        return 2
    elseif a == :S
        return 3
    elseif a == :W
        return 4
    elseif a == :grab
        return 5
    elseif a == :noop
        return 6
    else
        error("Invalid WumpusPOMDP action: $a")
    end
end

POMDPs.n_actions(pomdp::WumpusPOMDP) = 6

####
# OBSERVATION SPACE DEFINITION:
# All the used function are used in the same way as explained for state space definition
# POMDPs.observations(::MyPOMDP)
# POMDPs.obs_index(::MyPOMDP, ::Observation)
# POMDPs.n_observations(::MyPOMDP)
####

allObservations = [
    :breezeStenchGlitter
    :breezeStench
    :breezeGlitter
    :breeze
    :stenchGlitter
    :stench
    :glitter
    :none]

POMDPs.observations(pomdp::WumpusPOMDP) = allObservations
    
# There are 5 basic observations: breeze, stench, glitter
# The agent can recieve multiple observations at once (breeze&stench), 
# while pomdp model allows only one observation per time step. Hence the observations will be a combination


function POMDPs.obs_index(pomdp::WumpusPOMDP, o::Symbol)
    if o == :breezeStenchGlitter
        return 1
    elseif o == :breezeStench
        return 2
    elseif o == :breezeGlitter
        return 3
    elseif o == :breeze
        return 4
    elseif o == :stenchGlitter
        return 5
    elseif o == :stench
        return 6
    elseif o == :glitter
        return 7
    elseif o == :none
        return 8
    else
        error("Invalid WumpusPOMDP observation $o")
    end
end

POMDPs.n_observations(pomdp::WumpusPOMDP) = 8

####
# TRANSITION AND OBSERVATION DISTRIBUTION
# Transition and Observation models are intended to return a distribution of probabilities
# among possible states (in transition model) or observations (in observation model)
# These distributions indicate the probability of each result.
# In the example given three functions are used in order to solve this problem:
# (This is described for transition, but the concept holds for observation too)
#   Type DistributionType
#       Should contain an array of the states and an array of floats to indicate the probability of each state)
#   POMDPs.pdf(::DistributionType, ::Type)
#       Type can be ::State or ::Observation depending on the distribution we are evaluating
#       It is used in order to tell the program how to query the distribution we created the step before
#   POMDPs.iterator(::DistributionType)
#       Indicates which are the values which act as 'index' of the distribution we created before
#   POMDPs.rand(rng::AbstractRNG, od::ObservationDistribution)
#       ? Didn't really understand how to use this function. the description given is 
#       "Sampling function for use in simulation"
#
# POMDPs.pdf and POMDPs.rand can be substituted by the use of SparseCat
# Source code:
#   https://github.com/JuliaPOMDP/POMDPToolbox.jl/blob/master/src/distributions/sparse_cat.jl
# By looking at the source code we note how the previous POMDPs.pdf and POMDPs.rand are already defined for the SparseCat object
# In particular, by checking at l:43 we see that we only need to specify the probability of the ::State or ::Observation we want to be 
# different from 0, If we query about an object not present in the distribution, the probability returned will be 0
####

# Transition
function WumpusTransitionDistribution(s::WumpusState)
    return SparseCat([s], [1.0])
end

POMDPs.iterator(d::SparseCat{WumpusState, Float64}) = allStates

# Observation
function WumpusObservationDistribution(o::Symbol)
    SparseCat([o], [1.0])
end

POMDPs.iterator(d::SparseCat{Symbol, Float64}) = allObservations

####
# TRANSITION MODEL DEFINITION
# POMDPs.transition(::MyPOMDP, ::State, ::Action)
#   Define the transition model of the problem. It must return a probability distribution among the states in the way it is decribed before
#   ::State The state I wan when I took the action
#   ::Action The action I took
####
function POMDPs.transition(pomdp::WumpusPOMDP, s::WumpusState, a::Symbol)
    nextX = s.xA
    nextY = s.yA
    if a == :N
        # Check if we can move
        if s.yA < 4
            nextY = s.yA + 1
            next = WumpusStateMove(s, nextX, nextY)
            return WumpusTransitionDistribution(next)
        else
            return WumpusTransitionDistribution(s)
        end
    elseif a == :E
        # Check if we can move
        if s.xA < 4
            nextX = s.xA + 1
            next = WumpusStateMove(s, nextX, nextY)
            return WumpusTransitionDistribution(next)
        else
            return WumpusTransitionDistribution(s)
        end
    elseif a == :S
        # Check if we can move
        if s.yA > 1
            nextY = s.yA - 1
            next = WumpusStateMove(s, nextX, nextY)
            return WumpusTransitionDistribution(next)
        else
            return WumpusTransitionDistribution(s)
        end
    elseif a == :W
        # Check if we can move
        if s.xA > 1
            nextX = s.xA - 1
            next = WumpusStateMove(s, nextX, nextY)
            return WumpusTransitionDistribution(next)
        else
            return WumpusTransitionDistribution(s)
        end
    elseif a == :grab
        grab = false
        if (s.xA == s.xGold) && (s.yA == s.yGold)
            grab = true
        end
        next = WumpusStateGrab(s, grab)
        return WumpusTransitionDistribution(next)
    elseif a == :noop
        return WumpusTransitionDistribution(s)
    else
        error("Transition model: action $a is not defined")
    end
end

####
# OBSERVATION MODEL DEFINITION
# POMDPs.observation(::MyPOMDP, ::Action, ::State)
#   Define the observation model of the problem. It must return a probability distribution among the observations 
#   in the way it is decribed before
#   ::Action The last action I took
#   ::State The state I ended after taking the action
####
function POMDPs.observation(pomdp::WumpusPOMDP, a::Symbol, sp::WumpusState)
    # Flags for various observations:
    breeze = false
    stench = false
    glitter = false
    scream = false
    bump = false
    
    if isStateBreeze(sp)
        breeze = true
    end
    if isStateStench(sp)
        stench = true
    end
    if isStateGlitter(sp)
        glitter = true
    end
    
    
    # Check the various flags and set the various observations
    if breeze
        if stench
            if glitter
                return WumpusObservationDistribution(:breezeStenchGlitter)
            else
                return WumpusObservationDistribution(:breezeStench)
            end
        else
            if glitter
                return WumpusObservationDistribution(:breezeGlitter)
            else
                return WumpusObservationDistribution(:breeze)
            end
        end
    else
        if stench
            if glitter
                return WumpusObservationDistribution(:stenchGlitter)
            else
                return WumpusObservationDistribution(:stench)
            end
        else
            if glitter
                return WumpusObservationDistribution(:glitter)
            else
                return WumpusObservationDistribution(:none)
            end
        end
    end
    error("Observation Model: Error for action $a in state $s")
end

####
# REWARD FUNCTION DEFINITION
# POMDPs.reward(::MyPOMDP, ::State, ::Action)
#   Indicates the reward by taking
#   ::Action The action
#   ::State The state I was when I took the action
####
function POMDPs.reward(pomdp::WumpusPOMDP, s::WumpusState, a::Symbol)
# s the state where I was when I took the action
    if isPit(s, s.xA, s.yA)
        # If we are in a pit, no matter the action we take, it will always be bad
        return pomdp.r_inPit
    elseif (s.xA == s.xW) && (s.yA == s.yW)
        # If we have been eaten by the wumpus, no matter the action we take, it will always be bad
        return pomdp.r_inWumpusMouth
    elseif a == :grab
        # If we are grabbing the gold
        if (s.xA == s.xGold && s.yA == s.yGold) && !s.grabbed
            return pomdp.r_grabGold
        else
            return pomdp.r_action
        end
    elseif a == :noop
        return 0.0
    elseif a == :N || a == :E || a == :S || a == :W
        # Reward for moving around
        return pomdp.r_action
    else
        # We defined all the possible actions we could take
        error("Reward: action $a not defined for state $s")
    end
end

####
# USEFUL FUNCTIONS
# probability_check(::MyPOMDP)
#	Is used in order to check both the observation and transition model to return sound values.
#	If we want to check only one of the two models we can use
#	obs_prob_consistency_check(::MyPOMDP)
#	trans_prob_consistency_check(::MyPOMDP)
####
if checks_enabled
    println("Checks enabled!")
    wumpus = WumpusPOMDP()
    println("Checking observation probability..")
    obs_prob_consistency_check(wumpus)
    println("Completed")
    println("Checking transition probability..")
    trans_prob_consistency_check(wumpus)
    println("Completed")
    println("Performing overall check..")
    probability_check(wumpus)
    println("Completed")
    readline()
end

####
# SOLVE THE PROBLEM
# We use the SARSOP solver in order to generate a policy for the problem.
#   SARSOP.SarsopSolver(): Check the documentation for more parameters
#   https://juliapomdp.github.io/SARSOP.jl/latest/lib/api/#Solver-1
# After this we just need to use
#   POMDPs.solve(::MyPOMDP, ::Solver)
# to have a policy returned.
####
if load_problem
    println("Loading policy..")
    evaluated_problem = SARSOP.POMDPFile("wumpus.pomdpx")
    evaluated_policy = SARSOP.POMDPPolicy(WumpusPOMDP(), "wumpus.policy")
    evaluated_policy.alphas = SARSOP.POMDPAlphas(evaluated_policy.filename) # The problem is here, it loads a policy with no alphas
    println("Completed!")
else
    println("Evaluating policy..")
    solver = SARSOP.SARSOPSolver(fast=true, precision=5.0)
    policy = SARSOP.POMDPPolicy(WumpusPOMDP(), "wumpus.policy")
    evaluated_policy = POMDPs.solve(solver, WumpusPOMDP(), policy, pomdp_file_name="wumpus.pomdpx")
    evaluated_problem = SARSOP.POMDPFile("wumpus.pomdpx")
    println("Completed!")
end


####
# SIMULATION
# In order to simulate the problem we need
# POMDPs.updater(::Policy)
#   This is the belief updater for the problem. There are a series of default updaters in POMDPs.jl package
# POMDPToolbox.HistoryRecorder()
#   This is used in order to record the history during the simulation. Check the documentation for more parameters
# POMDPs.simulate(::HistoryRecorder, ::MyPOMDP, ::Policy, ::Updater, ::Belief_state)
#   Simulate the problem by following the given policy, and updating the initial given belief state
# After simulating the state, an object representing the history is returned.
# Example usage:
#   for (s, b, a, r, sp, op) in history
# Values for the tuple are:
#   s::State The previous state
#   b::Belief_state
#   a::Action
#   r::Reward
#   sp::State The resulting state
#   op::Observation
####
if sarsop_simulation
    println("SARSOP Simulation..")
    simulator = SARSOP.SARSOPSimulator(50, n_runs, output_file="SARSOP_sim.txt")
    SARSOP.simulate(simulator, evaluated_problem, evaluated_policy)
    println("Completed!")
    println("SARSOP Evaluation..")
    simulator = SARSOP.SARSOPEvaluator(50, n_runs, output_file="SARSOP_eval.txt")
    SARSOP.evaluate(simulator, evaluated_problem, evaluated_policy)
    println("Completed!")
    readline()
end

if normal_simulation
    println("Simulation..")
    #updater = SARSOP.updater(evaluated_policy)
    historyRecorder = HistoryRecorder(max_steps=50, rng=MersenneTwister(1))

    Totals = 0
    println("Simulating $n_runs runs")
    for i = 1:n_runs
        simulate_problem = WumpusPOMDP()
        history = simulate(historyRecorder, WumpusPOMDP(), evaluated_policy)
        run_total = 0
        println("Run $i:")
        j = 1
        if show_normal_simulation
            for (s, b, a, r, sp, op) in history
            # from state s to state sp through action a i got observation op
                println("Step $j:")
                println("from")
                println("$s")
                println("to")
                println("$sp")
                println("took action: $a, and recieved observation: $op")
                println("Reward for state/action pair: $r")
                run_total += r
                j += 1
            end
        end
        println("Total reward for run $i: $run_total")
        Totals += run_total
    end
    println("Completed!")
    Average = Totals/n_runs
    println("Runs: $n_runs - Average reward: $Average")
end

if generate_graph
    println("Generating graph..")
    graphgen = SARSOP.PolicyGraphGenerator("Wumpus.dot")
    SARSOP.polgraph(graphgen, evaluated_problem, evaluated_policy)
    println("Completed! Graph can be viewed at")
    println("http://www.webgraphviz.com/")
end


