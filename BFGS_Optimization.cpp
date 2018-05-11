#include "BFGS_Optimization.h"

#define C1 1e-4
#define C2 0.9

#define MACHINEPRECISION 1.1e-16

/** ===== Print Position Near Time Cutoff ========================== */

#define TIME_CUTOFF_POSITION_PRINT 604300

/** ================================================================ */


/** ===== Set the initial Hessian matrix =========================== */

#define IDENTITY
//#define INVERSE_HESSIAN

/** ================================================================ */


/** ===== Choose Finite Difference Formula For Gradient ============ */

#define FORWARD_DIFFERENCE
//#define CENTRAL_DIFFERENCE

/** ================================================================ */


/** ===== Print Step Monitor ======================================= */

#define PRINT_STEP_MONITOR

/** ================================================================ */


/** ===== Seed Random Number Generator ============================= */

#define SEED_RANDOM_NUMBER_GENERATOR

/** ================================================================ */


/** ===== Set the Maximum Number of Zoom Iterations in Line Search = */

#define ZOOM_GUARD 50

/** ================================================================ */

/** ===== Set the Maximum Number of Iterations ===================== */

#define MAX_ITERATIONS 6000

/** ================================================================ */


/** ===== Turn on Optimization Warning Messages ==================== */

#define OPT_WARNING_MESSAGE

/** ================================================================ */



double BFGS_Optimization::alpha(){

    alpha0 = 0.0;
    alpha1 = 1.0;
    alpha2 = alphaMax;

    phi0 = stepMonitor;
    phiPrime0 = gradient.transpose() * p;

    #ifdef PRINT_STEP_MONITOR

        if(phiPrime0 > 0.0){

            std::cout << "Positive directional derivative in line search..." << std::endl;

        }

    #endif // PRINT_STEP_MONITOR

    phi1 = phi(alpha1);

    if(phi1 > phi0 + C1 * alpha1 * phiPrime0){

        return zoom(alpha0,alpha1,phi0,phi1,phiPrime0);

    }

    phiPrime1 = phiPrime(alpha1);

    if(std::abs(phiPrime1) <= -C2 * phiPrime0){

        return alpha1;

    }

    if(phiPrime1 >= 0){

        return zoom(alpha1,alpha0,phi1,phi0,phiPrime0);

    }

    phi2 = phi(alpha2);

    if(phi2 > phi0 + C1 * alpha2 * phiPrime0 || phi2 >= phi1){

        return zoom(alpha1,alpha2,phi1,phi2,phiPrime1);

    }

    phiPrime2 = phiPrime(alpha2);

    if(std::abs(phiPrime2) <= -C2 * phiPrime0){

        return alpha2;

    }

    if(phiPrime2 >= 0){

        return zoom(alpha2,alpha1,phi2,phi1,phiPrime1);

    }

    #ifdef PRINT_STEP_MONITOR

        std::cout << "This iteration of BFGS has taken the maximum step size." << std::endl;

    #endif // PRINT_STEP_MONITOR

    maxStepSize = true;

    return alpha2;

}

void BFGS_Optimization::setAlphaJ(double& alphaj,double& alphaLow,double& alphaHigh,double& phiLow,double& phiHigh,double& phiLowPrime){



    if(alphaLow < alphaHigh){

        alphaj = alphaHigh * alphaHigh * phiLowPrime - alphaLow * (2.0 * phiHigh - 2.0 * phiLow + alphaLow * phiLowPrime);

        if((-phiHigh + phiLow + (alphaHigh - alphaLow) * phiLowPrime) != 0.0){

            alphaj /= 2.0 * (-phiHigh + phiLow + (alphaHigh - alphaLow) * phiLowPrime);

        }

        else assert(alphaj == 0.0);

        secondDerivativeTest = phiHigh - phiLow + (alphaLow-alphaHigh) * phiLowPrime;

        if((alphaHigh-alphaLow) * (alphaHigh-alphaLow) != 0.0){

            secondDerivativeTest /= (alphaHigh-alphaLow) * (alphaHigh-alphaLow);

        }

        else assert(std::abs(secondDerivativeTest) <= 100 * MACHINEPRECISION);

    }

    else{

        alphaj = alphaLow * alphaLow * phiLowPrime - alphaHigh * (2.0 * phiLow - 2.0 * phiHigh + alphaHigh * phiLowPrime);

        if((-phiLow + phiHigh + (alphaLow - alphaHigh) * phiLowPrime) != 0.0){

            alphaj /= 2.0 * (-phiLow + phiHigh + (alphaLow - alphaHigh) * phiLowPrime);

        }

        //else assert(alphaj == 0.0);

        secondDerivativeTest = phiLow - phiHigh + (alphaHigh-alphaLow) * phiLowPrime;

        if((alphaLow-alphaHigh) * (alphaLow-alphaHigh) != 0.0){

            secondDerivativeTest /= (alphaLow-alphaHigh) * (alphaLow-alphaHigh);

        }

        else assert(std::abs(secondDerivativeTest) <= 100 * MACHINEPRECISION);

    }


    return;

}

double BFGS_Optimization::zoom(double alphaLow,double alphaHigh,double phiLow,double phiHigh,double phiLowPrime){

    double alphaj,phij,phiPrimej;

    #ifdef PRINT_STEP_MONITOR

        std::cout << "Zooming into interval...\n";
        std::cout << alphaLow << "\t" << alphaHigh << std::endl;

    #endif // PRINT_STEP_MONITOR

    int zoomCounter = 0;

    while(true){

        zoomCounter++;

        if(zoomCounter > ZOOM_GUARD) {

            zoomGuard = true;
            return alphaj;

        }

        setAlphaJ(alphaj,alphaLow,alphaHigh,phiLow,phiHigh,phiLowPrime);

        if(secondDerivativeTest < 0.0){

            quadInterpolationFailure = true;

            if(phiHigh < phi0 + C1 * alphaHigh * phiPrime0){

                return alphaHigh;

            }

            else if(phiLow < phi0 + C1 * alphaLow * phiPrime0){

                return alphaLow;

            }

            else{

                wolfeConditionFailure = true;

                return alphaHigh;

            }

        }


        phij = phi(alphaj);

        #ifdef PRINT_STEP_MONITOR

            std::cout << "\t\t\t\t" <<  phij << "\t" << phiLowPrime  << ": \t" << alphaj << "\t" << alphaLow << "\t" << alphaHigh << "\t" << phiLow << "\t" << phiHigh << std::endl;

        #endif // PRINT_STEP_MONITOR

        if(phij > phi0 + C1 * alphaj * phiPrime0 || phij >= phiLow){

            alphaHigh = alphaj;
            phiHigh = phij;

        }

        else{

            phiPrimej = phiPrime(alphaj);



            if(std::abs(phiPrimej) <= -C2 * phiPrime0){

                return alphaj;

            }

            if(std::abs(phiPrimej) <= C2 * phiPrime0 && phiPrime0 >0){

                #ifdef PRINT_STEP_MONITOR

                    std::cout << "Unexpected Zoom Ending Condition... " << std::endl;

                #endif // PRINT_STEP_MONITOR

                return alphaj;

            }

            if(phiPrimej * (alphaHigh - alphaLow) >= 0.0){

                alphaHigh = alphaLow;
                phiHigh = phiLow;

            }

            alphaLow = alphaj;

            phiLow = phij;

        }

        if(alphaHigh < alphaLow){

            phiLowPrime = phiPrime(alphaHigh);

        }

        else{

            phiLowPrime = phiPrime(alphaLow);

        }

    }

}

double BFGS_Optimization::phiPrime(double& a){

    #ifdef FORWARD_DIFFERENCE

        double aPeps = a + eps;
        return (phi(aPeps)-phi(a))/eps;

    #endif // FORWARD_DIFFERENCE

    #ifdef CENTRAL_DIFFERENCE

        double aPeps = a + eps;
        double aMeps = a - eps;
        return (phi(aPeps)-phi(aMeps))/(2.0*eps);

    #endif // CENTRAL_DIFFERENCE

    #ifdef USE_GRAPE

        double aPeps = a + eps;
        return (phi(aPeps)-phi(a))/eps;

    #endif // USE_GRAPE

}

double BFGS_Optimization::phi(double& a){

    alphaPosition = position + a * p;

    return meritFunction.f(alphaPosition);

}

void BFGS_Optimization::printStepMonitor(){

    counter++;

    #ifdef PRINT_STEP_MONITOR

        std::cout << counter << "\t" << std::setprecision(16) << stepMonitor << "\t";
        std::cout << meritFunction.entropyMonitor() << "\t" << gradient.norm() << std::endl;

    #endif // PRINT_STEP_MONITOR

    #ifdef TIME_CUTOFF_POSITION_PRINT

        currentTime = omp_get_wtime();

        if( currentTime - startTime > TIME_CUTOFF_POSITION_PRINT ) savePosition();

    #endif // TIME_CUTOFF_POSITION_PRINT

    return;
}

void BFGS_Optimization::savePosition(){

    std::ofstream outfile("savedPosition.dat");

    outfile << "Running time: " << currentTime - startTime << std::endl << std::endl;

    for(int i=0;i<position.size();i++) outfile << std::setprecision(16) << position(i) << " ";
    outfile << std::endl << std::endl;

    outfile.close();

    return;

}

double BFGS_Optimization::minimize(){

    position = meritFunction.setInitialPosition();

    gradient.resize(meritFunction.funcDimension);

    funcDimension = meritFunction.funcDimension;

    stepMonitor = meritFunction.f(position);

    setGradient();

    setInitialH();

    I = Eigen::MatrixXd::Identity(meritFunction.funcDimension,meritFunction.funcDimension);

    alphaPosition.resize(meritFunction.funcDimension);

    counter = 0;

    zoomGuard = false;

    quadInterpolationFailure = false;

    wolfeConditionFailure = false;

    maxStepSize = false;

    maxIterationGuard = false;

    while(true){

        printStepMonitor();

        if(gradient.norm() < tol) break;

        if(counter > MAX_ITERATIONS) {

            maxIterationGuard = true;
            break;

        }

        p = - H * gradient;

        s = -position;

        y = -gradient;

        position += alpha() * p;

        stepMonitor =  meritFunction.f(position);

        if(isnan(stepMonitor)) break;
        else stepMonitorSafety = stepMonitor;

        setGradient();

        s += position;

        y += gradient;

        denom = y.transpose() * s;

        if(denom == 0.0) break;

        rho = 1.0 / denom;

        H = (I - rho * s * y.transpose()) * H * (I - rho * y * s.transpose()) + rho * s * s.transpose();

    }

    printResultReport();

    if(isnan(stepMonitor)) std::cout << "FIDELITY BEFORE NAN CRASH: " << stepMonitorSafety;

    if(stepMonitor < bestResult) bestResult = stepMonitor;

    return stepMonitor;

}


void BFGS_Optimization::setGradient(){

    #ifdef FORWARD_DIFFERENCE

        for(int i=0;i<meritFunction.funcDimension;i++){

            gradient(i)  = -stepMonitor;
            position(i) += eps;
            gradient(i) += meritFunction.f(position);
            position(i) -= eps;
            gradient(i) /= eps;

        }

    #endif // FORWARD_DIFFERENCE


    #ifdef CENTRAL_DIFFERENCE

        for(int i=0;i<meritFunction.funcDimension;i++){

            position(i) += eps;
            gradient(i) = meritFunction.f(position);
            position(i) -= 2*eps;
            gradient(i) -= meritFunction.f(position);
            position(i) += eps;
            gradient(i) /= 2*eps;

        }

    #endif // CENTRAL_DIFFERENCE


    return;

}


void BFGS_Optimization::printResultReport(){

    meritFunction.printReport(position);

    #ifdef OPT_WARNING_MESSAGE

        if(zoomGuard) std::cout << "Warning: Zoom Guard Was Hit..." << std::endl  <<  std::endl;

        if(quadInterpolationFailure) std::cout << "Warning: Quadratic Interpolation Failure..." << std::endl << std::endl;

        if(wolfeConditionFailure) std::cout << "Wolfe Condition failure on a step (should be rare)..." << std::endl << std::endl;

        if(maxStepSize) std::cout << "BFGS took the maximum step size for at least one iteration..." << std::endl << std::endl;

        if(maxIterationGuard) std::cout << "BFGS has taken the maximum number of steps imposed by the user..." << std::endl << std::endl;

    #endif // OPT_WARNING_MESSAGE

    return;

}



BFGS_Optimization::BFGS_Optimization(double tolerance,double maxStepSize,int intParam){

    #ifdef TIME_CUTOFF_POSITION_PRINT

        startTime = omp_get_wtime();

    #endif // TIME_CUTOFF_POSITION_PRINT

    #ifdef SEED_RANDOM_NUMBER_GENERATOR

        srand(611*time(NULL));

    #endif // SEED_RANDOM_NUMBER_GENERATOR

    #ifdef FORWARD_DIFFERENCE

        eps = sqrt(MACHINEPRECISION);

    #endif // FORWARD_DIFFERENCE

    #ifdef CENTRAL_DIFFERENCE

        eps = std::pow(MACHINEPRECISION,1.0/3.0);

    #endif // CENTRAL_DIFFERENCE

    tol = tolerance;

    alphaMax = maxStepSize;

    meritFunction.setMeritFunction(intParam);

    bestResult = 1e30;

}

void BFGS_Optimization::setInitialH(){

    #ifdef IDENTITY

        H = (1.0/(gradient.norm())) * Eigen::MatrixXd::Identity(meritFunction.funcDimension,meritFunction.funcDimension);

    #endif // IDENTITY

    #ifdef INVERSE_HESSIAN

        setInverseHessian();

    #endif // INVERSE_HESSIAN

    return;
}

void BFGS_Optimization::setInverseHessian(){

    Eigen::MatrixXd Hessian(meritFunction.funcDimension,meritFunction.funcDimension);

    // ToDo: If optimization just kind of sits around for a while near starting point, write an approximation to Hessian
    // see if that helps.

    assert(1>2 && "ToDo: write some code for approximating the inverse to the Hessian matrix");

    H = Hessian.inverse();

    return;

}

