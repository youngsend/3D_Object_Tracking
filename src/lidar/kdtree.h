/* \author Aaron Brown */

#include <vector>
#include <cmath>

// Structure to represent node of kd tree
struct Node
{
    LidarPoint point;
    int id;
    Node* left;
    Node* right;

    Node(LidarPoint arr, int setId)
            :	point(arr), id(setId), left(nullptr), right(nullptr)
    {}
};

struct KdTree
{
    Node* root;

    KdTree() : root(nullptr) {
    }

    static void insertHelper(Node **node, LidarPoint point, int id, int depth) {
        if (!(*node)) {
            *node = new Node(point, id);
        } else {
            // only compare x.
            if (point.x < ((*node)->point).x) {
                insertHelper(&((*node)->left), point, id, depth + 1);
            } else {
                insertHelper(&((*node)->right), point, id, depth + 1);
            }
        }
    }

    void insert(LidarPoint point, int id)
    {
        // Fill in this function to insert a new point into the tree
        // the function should create a new node and place correctly with in the root
        insertHelper(&root, point, id, 0);
    }

    static void searchHelper(Node *node, std::vector<int>& ids, LidarPoint target,
                             float distanceTol, int depth){
        if (node) {
            // only compare x.
            if (std::fabs(target.x-(node->point).x) <= distanceTol){
                ids.push_back(node->id);

                // need to search both left and right child because the box region cross both sides
                searchHelper(node->left, ids, target, distanceTol, depth+1);
                searchHelper(node->right, ids, target, distanceTol, depth+1);
            } else {
                // need only search one side
                if (target.x < (node->point).x) {
                    searchHelper(node->left, ids, target, distanceTol, depth+1);
                } else {
                    searchHelper(node->right, ids, target, distanceTol, depth+1);
                }
            }
        }
    }

// return a list of point ids in the tree that are within distance of target
    std::vector<int> search(LidarPoint target, float distanceTol)
    {
        std::vector<int> ids;
        searchHelper(root, ids, target, distanceTol, 0);
        return ids;
    }
};




