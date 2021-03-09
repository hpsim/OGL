 #!/bin/sh

# Sets a solver 
setSolver(){
    echo "Setting solver"
    baseFolder=$PWD
    cd $1
    sed "s/p{}/$2/g" system/fvSolutionTMP > system/fvSolution
    cd $baseFolder
}

# Derive a case from base case
deriveCase(){
    baseCase=boxTurbBase
    baseFolder=$PWD
    echo "Deriving case " $1 $2
    mkdir -p $1
    cp -r $baseCase $1/$2
    cd $baseFolder
}

setCells() {
    echo "Setting Cells $1"
    sed -i "s/16 16 16/$1 $1 $1/g" system/blockMeshDict
}

# Copy Pitzdaily base case
generateBaseCase() {
    baseCase=boxTurbBase
    FILE=$baseCase/system/controlDict
    baseFolder=$PWD
    if [ ! -f "$FILE" ]; then
        echo "Preparing base case"
        cp -r $FOAM_TUTORIALS/DNS/dnsFoam/boxTurb16  $baseCase

        echo "Meshing base case"
        cd $baseCase
        setCells $1
        blockMesh > blockMesh.log

        echo "add OGL.so"
        echo 'libs ("libOGL.so");' >> system/controlDict

        touch system/controlDict.foam

        # run only for 100 time steps
        endTime=0.1
        sed -i "s/endTime[ ]*[0-9.]*/endTime $endTime/g" system/controlDict

        # clear p solver settings
        sed -z 's/p\n[ ]*[{][^}]*[}]/p{}/g' system/fvSolution > system/fvSolutionTMP
    fi
    cd $baseFolder
}

runSolver() {
    solver=dnsFoam
    baseFolder=$PWD
    cd $1
    echo "Starting solver" $baseFolder
    $solver | tee solver.log | grep "Time ="
    tail -n 20 solver.log | grep "ExecutionTime" > performance.log
    cd $baseFolder
}

basePOFCG(){
    solver="OFCG"
    deriveCase "p" $solver
    setSolver  "p/$solver" "p{solver PCG;tolerance 1e-06; relTol 0.0;smoother none;preconditioner none;maxIter 10000;}"
    runSolver "p/$solver"
}

basePGKOCG(){
    solver="GKOCG"
    deriveCase "p" "$solver"
    setSolver  "p/$solver" "p{solver $solver;tolerance 1e-06; relTol 0.0;smoother none;preconditioner none;maxIter 10000;}"
    runSolver "p/$solver"
}

basePGKOCGOMP(){
    solver="GKOCGOMP"
    deriveCase "p" "$solver"
    setSolver  "p/$solver" "p{solver GKOCG;tolerance 1e-06; relTol 0.0;smoother none;preconditioner none;maxIter 10000;app_executor omp; executor omp;}"
    runSolver "p/$solver"
}

basePGKOCUDA(){
    solver="GKOCGCUDA"
    deriveCase "p" "$solver"
    setSolver  "p/$solver" "p{solver GKOCG;tolerance 1e-06; relTol 0.0;smoother none;preconditioner none;maxIter 10000; app_executor omp; executor cuda;}"
    runSolver "p/$solver"
}


generatePerformanceLog(){
    mkdir -p stats
    timestamp=$(date +%s)

    lscpu | grep "Modelname" > stats/stats$timestamp.txt
    lspci | grep "VGA" >> stats/stats$timestamp.txt
    git log | head -n 1 >>  stats/stats$timestamp.txt
    find . -name "performance.log" -exec echo {} \; -exec cat {} >> stats/stats$timestamp.txt  \;
}

# run symmetric solver
mkdir -p Tests

cd Tests

for number in 8 16 32 64 128
do
echo $number
    mkdir -p $number
    cd $number

    generateBaseCase $number
    basePGKOCGOMP
    basePOFCG
    basePGKOCG
    # basePGKOCUDA

    generatePerformanceLog

    cd ..
done

cd ..
exit 0

