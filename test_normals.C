#include "fvCFD.H"
#include "vector.H"

int main(int argc, char *argv[])
{
    // Initialize OpenFOAM case and environment
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    // Check if the patch name is provided as an argument
    if (argc < 2)
    {
        FatalErrorInFunction << "Usage: " << argv[0] << " patchName" << exit(FatalError);
    }

    // Get the patch name from command line arguments
    word patchName(argv[1]);

    // Find the patch by name in the mesh
    const polyBoundaryMesh& boundary = mesh.boundary();
    label patchID = boundary.findPatchID(patchName);

    if (patchID == -1)
    {
        FatalErrorInFunction << "Patch " << patchName << " not found in mesh." << exit(FatalError);
    }

    const fvPatch& patch = boundary[patchID];
    const labelList& patchPoints = patch.facePoints();
    
    // Loop through each point on the patch
    forAll(patchPoints, pointi)
    {
        label pointIndex = patchPoints[pointi];
        point pointP = mesh.points()[pointIndex];

        vector avgNormal(0, 0, 0); // To store the average normal vector
        int faceCount = 0;         // Count of faces containing point P

        // Loop over the faces of the patch
        const labelList& patchFaces = patch.faceCells();

        forAll(patchFaces, facei)
        {
            const face& f = mesh.faces()[patchFaces[facei]];

            if (f.containsPoint(pointIndex))
            {
                // This face contains point P, so we need to find the two connected points
                labelList facePoints = f.points();

                // Find two points on the face connected to P
                label p1 = -1, p2 = -1;
                for (int fp = 0; fp < facePoints.size(); ++fp)
                {
                    if (facePoints[fp] == pointIndex)
                    {
                        // The two neighboring points connected to P on this face
                        p1 = facePoints[(fp - 1 + facePoints.size()) % facePoints.size()];
                        p2 = facePoints[(fp + 1) % facePoints.size()];
                        break;
                    }
                }

                if (p1 == -1 || p2 == -1)
                {
                    FatalErrorInFunction << "Unable to find neighboring points for point " << pointIndex << exit(FatalError);
                }

                // Points connected to point P
                point point1 = mesh.points()[p1];
                point point2 = mesh.points()[p2];

                // Compute vectors connecting point P with point1 and point2
                vector v1 = point1 - pointP;
                vector v2 = point2 - pointP;

                // Compute the normal vector orthogonal to v1 and v2 (cross product)
                vector normal = Foam::cross(v1, v2);

                // Accumulate the normal vector
                avgNormal += normal;
                faceCount++;
            }
        }

        // Average the normal vector
        if (faceCount > 0)
        {
            avgNormal /= faceCount;

            // Normalize the vector
            avgNormal /= mag(avgNormal);

            // Output the normalized average normal vector for point P
            Info << "Point " << pointIndex << " avg normal: " << avgNormal << endl;
        }
        else
        {
            Info << "Point " << pointIndex << " not found in any faces." << endl;
        }
    }

    Info << "Completed normal vector calculation for patch: " << patchName << endl;

    return 0;
}

