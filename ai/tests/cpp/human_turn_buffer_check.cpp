// Appended to the actual dispatch/Act/queue source by test_human_turn_buffer.py.
// The surrounding probe supplies deterministic time and records executed turns.

static bool Press(ePlayer &player,gCycle &cycle,REAL time,uActionPlayer &action,REAL value=1)
{
    cycle.Advance(time);
    return player.Dispatch(&cycle,&action,value);
}

int main(int argc,char **argv)
{
    assert(argc==2);
    std::string scenario=argv[1];
    ePlayer player;
    NetPlayer peer;
    player.netPlayer=&peer;
    gCycle cycle;
    auto &left=eGameObject::se_turnLeft;
    auto &right=eGameObject::se_turnRight;

    if(scenario=="default-dispatch") {
        eGameObject other;
        assert(player.Dispatch(&other,&left,.75));
        assert(other.ordinaryCalls==1 && other.lastValue==.75 && peer.activity==1);
        assert(Press(player,cycle,0,left));
        assert(cycle.executed.size()==1 && cycle.executed[0].second==-1);
    }
    else if(scenario=="burst") {
        assert(Press(player,cycle,0,left));
        for(int i=1;i<=40;++i) assert(Press(player,cycle,.001*i,left));
        assert(cycle.pendingTurns.size()==1 && cycle.turnCalls==2);
        assert(cycle.executed.size()==1 && cycle.destinations==1);
        cycle.Advance(.101);
        assert(cycle.pendingTurns.empty() && cycle.executed.size()==2);
        assert(std::abs(cycle.executed[1].first-.1)<1e-9);
        cycle.Advance(.9);
        assert(cycle.executed.size()==2 && cycle.destinations==2);
    }
    else if(scenario=="opposite") {
        Press(player,cycle,0,left);
        Press(player,cycle,.01,left);
        assert(Press(player,cycle,.02,right));
        assert(cycle.pendingTurns.size()==1 && cycle.pendingTurns.front()==-1);
        cycle.Advance(.101);
        assert(cycle.executed.size()==2 && cycle.executed.back().second==-1);
        Press(player,cycle,.11,right);
        assert(cycle.pendingTurns.size()==1 && cycle.pendingTurns.front()==1);
        cycle.Advance(.201);
        assert(cycle.executed.size()==3 && cycle.executed.back().second==1);
    }
    else if(scenario=="cooldown-release") {
        Press(player,cycle,0,left);
        Press(player,cycle,.01,left);
        assert(!Press(player,cycle,.02,left,0));
        assert(!Press(player,cycle,.03,left,-1));
        assert(!Press(player,cycle,.04,left,.5));
        assert(!Press(player,cycle,.05,left,std::numeric_limits<REAL>::quiet_NaN()));
        assert(cycle.pendingTurns.size()==1 && cycle.executed.size()==1);
        cycle.Advance(.099);
        assert(cycle.executed.size()==1);
        cycle.Advance(.101);
        assert(cycle.executed.size()==2);
        cycle.Advance(1);
        assert(cycle.executed.size()==2);
    }
    else if(scenario=="braking") {
        Press(player,cycle,0,left);
        Press(player,cycle,.01,left);
        int const before=cycle.destinations;
        assert(Press(player,cycle,.02,gCycle::s_brake,1));
        assert(cycle.braking==1 && cycle.destinations==before+1);
        assert(Press(player,cycle,.03,gCycle::s_brake,0));
        assert(cycle.braking==0 && cycle.destinations==before+2);
        assert(cycle.pendingTurns.size()==1);
    }
    else if(scenario=="bot-isolation") {
        cycle.Act(&left,1);
        for(int i=0;i<12;++i) cycle.Act(&left,1); // neural/chat auto-drive bypass human route
        assert(cycle.pendingTurns.size()==12);
        gCycle server;
        server.Turn(-1);
        for(int i=0;i<12;++i) server.Turn(-1); // ordinary movement/destination caller
        assert(server.pendingTurns.size()==12);
    }
    else if(scenario=="classic") {
        sg_cycleInputBuffer=0;
        Press(player,cycle,0,left);
        for(int i=1;i<=12;++i) Press(player,cycle,.001*i,left);
        assert(cycle.pendingTurns.size()==12 && cycle.turnCalls==13);
        // CYCLE_TURN_MEMORY remains the legacy cancellation threshold, not a cap.
        Press(player,cycle,.02,right);
        assert(cycle.pendingTurns.size()==11);
    }
    else if(scenario=="existing-auto-drive") {
        cycle.Act(&left,1);
        for(int i=0;i<3;++i) cycle.Act(&left,1);
        auto const before=cycle.pendingTurns;
        int const calls=cycle.turnCalls;
        assert(Press(player,cycle,.02,right));
        assert(cycle.pendingTurns==before && cycle.turnCalls==calls);
    }
    else if(scenario=="camera-start-gates") {
        Camera camera;
        player.cam=&camera;
        camera.consumes=true;
        assert(Press(player,cycle,0,left));
        assert(cycle.turnCalls==0 && camera.calls==1);
        camera.consumes=false;
        assert(!Press(player,cycle,-.1,left));
        assert(cycle.turnCalls==0);
        assert(!player.Dispatch(nullptr,&left,1));
        timer.speed=0;
        assert(!Press(player,cycle,.01,left));
        assert(cycle.turnCalls==0);
    }
    else if(scenario=="per-cycle-lifetime") {
        Press(player,cycle,0,left);
        Press(player,cycle,.01,left);
        gCycle second;
        assert(Press(player,second,.02,right));
        assert(second.executed.size()==1 && second.executed[0].second==1);
        cycle.alive=false;
        cycle.Advance(.101);
        assert(cycle.executed.size()==1);
        gCycle respawn;
        assert(respawn.pendingTurns.empty());
        assert(Press(player,respawn,.2,left));
        assert(respawn.executed.size()==1 && respawn.pendingTurns.empty());
    }
    else if(scenario=="configured-cooldown") {
        turnDelay=.075;
        Press(player,cycle,0,left);
        Press(player,cycle,.01,right);
        for(int i=1;i<=20;++i) Press(player,cycle,.01+.001*i,right);
        cycle.Advance(.074);
        assert(cycle.executed.size()==1 && cycle.pendingTurns.size()==1);
        cycle.Advance(.076);
        assert(cycle.executed.size()==2 && cycle.pendingTurns.empty());
        assert(std::abs(cycle.executed.back().first-.075)<1e-9);
        cycle.Advance(.5);
        assert(cycle.executed.size()==2);
    }
    else return 2;
}
