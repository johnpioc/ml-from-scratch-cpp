#include <cstdlib>
#include <iostream>

// ===============================================================================================
// CONSTANTS
// ===============================================================================================
const int SUCCESS_NUM = 0;

const int USAGE_ERROR_NUM = 1;
const std::string USAGE_ERROR_MSG = "Usage Error";

enum ModelToRun { NONE, LINEAR_REGRESSION };

// ===============================================================================================
// FUNCTION DECLARATIONS
// ===============================================================================================
int parseCliArguments(int argc, char* argv[], ModelToRun& modelToRun);
void parseExitCode(int exitCode);

// ===============================================================================================
// MAIN FUNCTION
// ===============================================================================================
int main(int argc, char* argv[]) {
    // Parse CLI arguments
    ModelToRun modelToRun = NONE;
    parseExitCode(parseCliArguments(argc, argv, modelToRun));

    return 0;
}

// ===============================================================================================
// HELPERS
// ===============================================================================================
int parseCliArguments(int argc, char* argv[], ModelToRun& modelToRun) {
    // Skip program name
    argc--;
    argv++;

    // Initialise model to none
    modelToRun = NONE;

    // Iterate through every command line argument
    while (argc > 0) {
        std::string current(argv[0]);

        if (modelToRun != NONE) {
            return USAGE_ERROR_NUM;
        } else if (current == "1") {  // linear regression
            modelToRun = LINEAR_REGRESSION;
        } else {
            return USAGE_ERROR_NUM;
        }
    }

    if (modelToRun == USAGE_ERROR_NUM) return USAGE_ERROR_NUM;

    return SUCCESS_NUM;
}

void parseExitCode(int exitCode) {
    switch (exitCode) {
        case USAGE_ERROR_NUM: {
            std::cerr << USAGE_ERROR_MSG;
            break;
        }
        default:
            break;
    }

    if (exitCode != 0) std::exit(exitCode);
}
