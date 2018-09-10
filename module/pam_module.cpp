#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <security/pam_appl.h>
#include <security/pam_modules.h>

#define PAM_FACE_MODULE "/etc/pam_face_module"
#define FACE_MODULE_FEATURE_FILE "features.xml"

PAM_EXTERN int pam_sm_setcred( pam_handle_t * pamh, int flags, int argc, const char ** argv )
{
    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_acct_mgmt( pam_handle_t * pamh, int flags, int argc, const char ** argv )
{
    return PAM_SUCCESS;
}

/* expected hook, this is where custom stuff happens */
PAM_EXTERN int pam_sm_authenticate( pam_handle_t * pamh, int flags, int argc, const char ** argv )
{

    printf("pam_sm_authenticate ::::::!!\n");
    std::cout << "pam_sm_authenticate0000000 " << std::endl;
    std::cerr << "pam_sm_authenticate0000 " << std::endl;


    // First verify we can get username
    const char * user;
    int ret = pam_get_user( pamh, &user, "Username: " );
    if ( ret != PAM_SUCCESS )
    {
            std::cout << "pam_sm_authenticate1 " << std::endl;
    std::cerr << "pam_sm_authenticate1 " << std::endl;
        return ret;
    }
    std::string username( user );


    if (FILE *file = fopen("/home/xavier/Bureau/toto.txt", "r")) {
        fclose(file);
            std::cout << "pam_sm_authenticate2 " << std::endl;
    std::cerr << "pam_sm_authenticate2 " << std::endl;
        return PAM_SUCCESS;
    } else {
            std::cout << "pam_sm_authenticate3 " << std::endl;
    std::cerr << "pam_sm_authenticate3 " << std::endl;
        return PAM_AUTH_ERR;
    }
}

