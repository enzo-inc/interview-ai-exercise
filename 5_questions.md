# Question 1
How do you authenticate to the StackOne API?

## C0
### Answer
You authenticate to the StackOne API by calling the POST /connect_sessions/authenticate endpoint with a JSON request body matching the ConnectSessionAuthenticate schema.

Endpoint: POST /connect_sessions/authenticate
Body (JSON): parameters required to authenticate the connect session (as defined in ConnectSessionAuthenticate)
Successful response: HTTP 201 with a ConnectSession object containing the details of the authenticated session
Error responses:
400 – invalid request
401 – unauthorized access
403 – forbidden
In practice, your client/application sends the appropriate authentication parameters in the body of this request; on success, you use the returned connect session details for subsequent authorized operations.

### Sources
stackone.json - Path: /connect_sessions/authenticate
ats.json - Schema: schemas
stackone.json - Path: /accounts/{id}
crm.json - Schema: schemas

## C1
### Answer
You authenticate to the StackOne API using HTTP Basic authentication.

For each request (including POST /connect_sessions/authenticate, GET /accounts/{id}, and POST /unified/proxy), you must include an Authorization header with your StackOne API credentials encoded in Basic auth format, for example:

Authorization: Basic <base64_encoded_api_key_and_secret>

(Where <base64_encoded_api_key_and_secret> is the Base64-encoded string of username:password or api_key:secret, as provided by StackOne.)

### Sources
stackone.json - Path: /connect_sessions/authenticate
stackone.json - Schema: ConnectSessionAuthenticate
stackone.json - Path: /accounts/{id}
stackone.json - Path: /unified/proxy

## C2
### Answer
You authenticate to the StackOne API using HTTP Basic Authentication.

When calling endpoints like:

POST /connect_sessions/authenticate
POST /connect_sessions
GET /accounts/{id}
you must include an Authorization header in this form:

Authorization: Basic <base64(username:password)>

Where:

username = your StackOne API key (or provided username)
password = your StackOne API secret (or provided password)
base64(username:password) is the Base64 encoding of the string username:password.
Example curl:

curl -X POST https://api.stackone.com/connect_sessions/authenticate \
  -u "YOUR_API_KEY:YOUR_API_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"token": "CONNECT_SESSION_TOKEN"}'

The -u "user:pass" flag in curl automatically sets the correct HTTP Basic Authorization header.

### Sources
stackone.json - Path: /connect_sessions/authenticate
stackone.json - Schema: ConnectSessionAuthenticate
stackone.json - Path: /connect_sessions
stackone.json - Path: /accounts/{id}

## C3
### Answer
You authenticate to the StackOne API using HTTP Basic Authentication.

Every endpoint shown (POST /connect_sessions/authenticate, POST /connect_sessions, GET /accounts/{id}) specifies:

Authentication: HTTP Basic

So when calling these endpoints, include an Authorization header with your StackOne API credentials encoded in Basic auth format, for example:

Authorization: Basic base64(username:password)

(How you obtain the actual username/password or API key is determined by your StackOne account setup; once you have them, you send them via HTTP Basic on each request.)

### Sources
stackone.json - Path: /connect_sessions/authenticate
stackone.json - Schema: ConnectSessionAuthenticate
stackone.json - Path: /connect_sessions
stackone.json - Path: /accounts/{id}

## C4
### Answer
You authenticate to the StackOne API using HTTP Basic Authentication on all the endpoints shown (e.g. /connect_sessions, /connect_sessions/authenticate, /accounts).

In practice, that means:

Send an Authorization header on each request:
Authorization: Basic <base64-encoded-credentials>
The credentials (before base64-encoding) are typically:
api_key: (API key as the username and an empty password), or
username:password (if your account uses explicit user/password credentials)
Example (pseudo‑curl):

curl -X POST https://api.stackone.com/connect_sessions \
  -u "YOUR_API_KEY:" \
  -H "Content-Type: application/json" \
  -d '{
    "origin_owner_id": "123",
    "origin_owner_name": "Acme Inc."
  }'

The -u "YOUR_API_KEY:" flag sets the HTTP Basic Authorization header.

### Sources
stackone.json - Path: /connect_sessions/authenticate
stackone.json - Path: /connect_sessions
stackone.json - Schema: ConnectSessionAuthenticate
stackone.json - Schema: ConnectSessionCreate
stackone.json - Path: /accounts

## C5
### Answer
You authenticate to the StackOne API using HTTP Basic authentication.

The provided endpoints (POST /connect_sessions and POST /connect_sessions/authenticate) both specify:

Authentication: HTTP Basic
No further details (such as the exact username/password format or API key usage) are given in the provided context.

### Sources
stackone.json - Path: /connect_sessions/authenticate
stackone.json - Schema: ConnectSessionAuthenticate
stackone.json - Path: /connect_sessions
stackone.json - Schema: ConnectSessionCreate
stackone.json - Schema: AuthenticationMetaItem

# Question 2
Can I retrieve all linked accounts with workday provider?

## C0
### Answer
Use the GET /accounts endpoint and filter by provider using the (now deprecated) provider query parameter or the newer providers list:

Deprecated but still present:
GET /accounts?provider=workday

Recommended (uses the array-based filter):
GET /accounts?providers[]=workday

This will return only the linked accounts whose provider is Workday

### Sources
stackone.json - Path: /accounts
hris.json - Schema: schemas
ats.json - Schema: schemas
stackone.json - Schema: schemas

## C1
### Answer
Yes.

Use the STACKONE List Accounts endpoint and filter by the Workday provider:

Endpoint

GET /accounts

Key query parameter

provider=workday
or, if you want to include Workday in a list of providers:
providers[]=workday
Example

GET /accounts?provider=workday&page=1&page_size=50 HTTP/1.1
Authorization: Basic <credentials>

This will return all linked accounts whose provider is workday.

### Sources
stackone.json - Path: /accounts
hris.json - Path: /unified/hris/employees/{id}/work_eligibility/{subResourceId}
hris.json - Path: /unified/hris/locations/{id}
hris.json - Path: /unified/hris/companies/{id}

## C2
### Answer
Yes.

Use the STACKONE – List Accounts endpoint and filter by provider:

Endpoint

GET /accounts
Key query parameter

provider=workday
Example request

GET /accounts?provider=workday HTTP/1.1
Authorization: Basic <base64-credentials>
Host: api.stackone.com

You can also paginate with page and page_size, and if you need multiple providers at once you could instead use providers[]=workday.

### Sources
stackone.json - Path: /accounts
hris.json - Path: /unified/hris/employees/{id}/work_eligibility/{subResourceId}

## C3
### Answer
Yes. Use the GET /accounts endpoint and filter by the Workday provider.

Example request:

GET /accounts?provider=workday HTTP/1.1
Host: api.stackone.com
Authorization: Basic <your-credentials>

If your API expects the provider in the providers array instead, you can do:

GET /accounts?providers[]=workday HTTP/1.1
Host: api.stackone.com
Authorization: Basic <your-credentials>

Both return a paginated list of LinkedAccount objects where provider is Workday. Use page and page_size if you need to page through all results.

### Sources
stackone.json - Path: /accounts
stackone.json - Path: /accounts/{id}
stackone.json - Schema: LinkedAccount
stackone.json - Schema: LinkedAccountMeta

## C4
### Answer
Yes. Use the GET /accounts endpoint with the provider (or providers) query parameter set to workday.

Example:

GET /accounts?provider=workday
Authorization: Basic <credentials>

or, if you prefer the array-style filter (depending on how your client encodes arrays):

GET /accounts?providers=workday
Authorization: Basic <credentials>

This will return all linked accounts whose provider is Workday, in the LinkedAccount schema format. You can also add page and page_size for pagination if needed.

### Sources
stackone.json - Path: /accounts
stackone.json - Schema: LinkedAccount
stackone.json - Path: /accounts/{id}
stackone.json - Path: /accounts/{id}/meta

## C5 
### Answer
Yes. You can list all linked accounts for the Workday provider using the GET /accounts endpoint and the provider query parameter.

Example request:

GET /accounts?provider=workday
This will return an array of LinkedAccount objects filtered to those whose provider is workday.
### Sources
stackone.json - Path: /accounts
stackone.json - Schema: LinkedAccount
stackone.json - Path: /accounts/{id}
stackone.json - Path: /accounts/{id}/meta

# Question 3
What is the default expiry of the session token?

## C0
### Answer
The default expiry of the session token is 30 minutes.

### Sources
stackone.json - Schema: schemas
stackone.json - Path: /connect_sessions
stackone.json - Path: /mcp

## C1
### Answer
The default expiry of the session token is 1 hour (3600 seconds).

### Sources
stackone.json - Path: /connect_sessions
stackone.json - Path: /connect_sessions/authenticate
stackone.json - Schema: ConnectSessionAuthenticate

## C2
### Answer
The provided context does not specify any default expiry duration for the session token.

### Sources
stackone.json - Path: /connect_sessions
stackone.json - Path: /connect_sessions/authenticate
stackone.json - Schema: ConnectSessionCreate

## C3
### Answer
The default expiry of the session token is 24 hours (86,400 seconds).

### Sources
stackone.json - Path: /connect_sessions
stackone.json - Path: /connect_sessions/authenticate
stackone.json - Schema: ConnectSessionCreate

## C4
### Answer
The default expiry of the session token is 1 hour (3600 seconds).

### Sources
stackone.json - Path: /connect_sessions
stackone.json - Schema: ConnectSessionCreate
stackone.json - Path: /connect_sessions/authenticate

## C5 
### Answer
I don't have enough information in the StackOne documentation to answer this question. The context only states that expires_in is a field indicating how long the session is valid in seconds, but it does not specify what the default value is if expires_in is not provided.

### Sources
stackone.json - Path: /connect_sessions
stackone.json - Schema: ConnectSessionCreate
stackone.json - Schema: ConnectSessionTokenAuthLink

# Question 4
What fields must be sent to create a course on an LMS?

## C0
### Answer
To create a course on an LMS with this API, only the field specified as required in the schema must be sent:

learning_object_external_reference
All other listed fields for courses (such as title, description, languages, cover_url, url, active, external_reference, content_ids, remote_content_ids, unified_custom_fields, etc.) are marked as nullable/optional and are not strictly required to create the course.

### Sources
lms.json - Path: /unified/lms/courses
lms.json - Schema: schemas

## C1
### Answer
The provided context only documents:

The Course schema (fields that exist on a Course object), and
GET endpoints to list and retrieve courses.
There is no create (POST) endpoint nor any documentation of required fields to create a course in this snippet. Because of that, we cannot determine from this context which fields are mandatory when creating a course on an LMS via this API.

From the schema, we can see what a Course can contain (e.g., title, description, languages, cover_url, url, active, duration, categories, skills, etc.), but the context doesn’t specify:

a POST /unified/lms/courses endpoint, and
which of these properties are required vs. optional in a create request.
So, based solely on the given context, no specific set of “must send” fields can be identified for creating a course.

### Sources
lms.json - Schema: Course
lms.json - Path: /unified/lms/courses/{id}
lms.json - Path: /unified/lms/courses

## C2
### Answer
Based on the provided context, the Course schema lists many possible properties, but it does not specify which of them are required when creating a course. It only states that:

“Partial updates are not supported. When updating content, you must provide all the same fields that are required when creating content.”

However, the only creation/upsert endpoint shown (PUT /unified/lms/content, lms_upsert_content) is for external linking learning objects (content), not for courses, and the required fields for that endpoint are not included in the excerpt.

For courses, the context only provides:

The Course schema (with fields like id, remote_id, title, description, content_ids, etc.).
Read-only endpoints:
GET /unified/lms/courses (list courses)
GET /unified/lms/courses/{id} (get a specific course)
There is no course creation endpoint (e.g. POST /unified/lms/courses) and no indication of which Course fields are mandatory on creation.

Therefore, from the information given, it is not possible to determine which fields must be sent to create a course on an LMS. The required fields for course creation are not specified in the provided context.

### Sources
lms.json - Schema: Course
lms.json - Path: /unified/lms/content
lms.json - Path: /unified/lms/courses
lms.json - Path: /unified/lms/courses/{id}
lms.json - Schema: CourseResult

## C3
### Answer
Based on the provided context, there is no endpoint described for creating or updating a Course object (only for listing and retrieving courses). The only write endpoint shown is:

PUT /unified/lms/content (for external linking learning objects, not Courses), and it notes that partial updates are not supported but does not list which fields are required for creation.
Because:

No POST /unified/lms/courses or PUT /unified/lms/courses/{id} endpoint is documented.
No “required” attributes are marked on the Course schema properties.
The creation/upsert endpoint shown (lms_upsert_content) is for a different schema (“content”, not “course”), and even there the required fields are not specified.
It is not possible, from this context alone, to determine which fields are mandatory to create a course on the LMS. The schema lists all possible properties of a Course (e.g. title, description, languages, url, active, etc.), but none are explicitly marked as required for creation.

To know the exact required fields, you would need either:

The documentation for the course creation endpoint (e.g. POST /unified/lms/courses) including its “required” markers, or
Provider-specific documentation that states which fields must be supplied when creating a course.

### Sources
lms.json - Schema: Course
lms.json - Path: /unified/lms/courses
lms.json - Path: /unified/lms/courses/{id}
lms.json - Path: /unified/lms/content
lms.json - Schema: CourseResult

## C4
### Answer
Based on the provided context, no required fields for creating a course are specified—the schema and operation you shared (lms_get_course) only describe:

The Course schema (what a course object looks like), and
A GET operation to retrieve a course by ID.
There is no "create course" (POST) endpoint or request body schema described here, so we cannot infer which fields are mandatory for creation. The listed fields (e.g., title, description, languages, url, active, etc.) are properties that may exist on a course object, but the context does not state which of them are required in a create request.

So, with the information given:

We cannot determine any mandatory fields for creating a course.
You would need the documentation for the course creation endpoint (likely a POST to /courses or similar), which should explicitly mark which fields are required in the request body.

### Sources
lms.json - Schema: Course
lms.json - Schema: CourseResult
lms.json - Path: /unified/lms/courses/{id}

## C5 
### Answer
I don't have enough information in the StackOne documentation to answer this question. The context only describes the Course and CourseResult schemas and the lms_get_course (GET) operation for retrieving a course. It does not document a create course endpoint or specify which fields are required when creating a course on an LMS.

### Sources
lms.json - Schema: Course
lms.json - Schema: CourseResult
lms.json - Path: /unified/lms/courses/{id}

# Question 5
What is the response body when listing an employee?

## C0
### Answer
The response body when listing an employee is an object matching the schema:

#/components/schemas/EmployeesPaginated

### Sources
hris.json - Path: /unified/hris/employees/{id}/employments/{subResourceId}
hris.json - Path: /unified/hris/employees/{id}/employments
hris.json - Path: /unified/hris/employees/{id}
hris.json - Path: /unified/hris/employees

## C1
### Answer
When listing an employee, the response body (HTTP 200) is:

{
  "data": {
    "id": "string",                     // Unique identifier
    "remote_id": "string",              // Provider's unique identifier
    "unified_custom_fields": {},        // Custom Unified Fields
    "title": "string",                  // e.g. "Mr", "Ms", "Dr"
    "first_name": "string",
    "last_name": "string",
    "name": "string",                   // Full name
    "display_name": "string",
    "avatar_url": "string",
    "personal_email": "string",
    "personal_phone_number": "string",
    "work_email": "string",
    "work_phone_number": "string"
    // ...other employee fields if defined in the full spec
  }
}

### Sources
hris.json - Path: /unified/hris/employees/{id}
hris.json - Path: /unified/hris/employees/{id}/tasks/{subResourceId}
hris.json - Path: /unified/hris/employees/{id}/employments/{subResourceId}
hris.json - Path: /unified/hris/employees/{id}/tasks
hris.json - Path: /unified/hris/employees/{id}/skills

## C2
### Answer
{
  "data": {
    "id": "string",                    // Unique identifier
    "remote_id": "string",             // Provider's unique identifier
    "unified_custom_fields": {},       // Custom Unified Fields for your StackOne project
    "title": "string",                 // Prefix (e.g., Mr, Ms, Dr)
    "first_name": "string",            // Employee first name
    "last_name": "string",             // Employee last name
    "name": "string",                  // Full name
    "display_name": "string",          // Display name
    "avatar_url": "string",            // Avatar URL
    "personal_email": "string",        // Personal email
    "personal_phone_number": "string", // Personal phone number
    "work_email": "string",            // Work email
    "work_phone_number": "string"      // Work phone number
    // (truncated fields if any are not shown in the context)
  }
}

### Sources
hris.json - Path: /unified/hris/employees/{id}
hris.json - Path: /unified/hris/employees/{id}/tasks/{subResourceId}
hris.json - Path: /unified/hris/employees/{id}/employments/{subResourceId}

## C3
### Answer
hen listing an employee, the response body is:

next (string)
data (array of object) (required)
raw (array of object)

### Sources

hris.json - Path: /unified/hris/employees/{id}
hris.json - Path: /unified/hris/employees/{id}/tasks/{subResourceId}
hris.json - Path: /unified/hris/employees/{id}/employments/{subResourceId}
hris.json - Path: /unified/hris/employees/{id}/tasks
hris.json - Path: /unified/hris/employees/{id}/skills

## C4
### Answer
When listing an employee, the successful response (HTTP 200) has this structure:

{
  "data": {
    "id": "string",                     // Unique identifier
    "remote_id": "string",              // Provider's unique identifier
    "unified_custom_fields": {},        // Custom Unified Fields
    "title": "string",                  // Prefix (e.g., Mr, Ms, Dr)
    "first_name": "string",
    "last_name": "string",
    "name": "string",                   // Full name
    "display_name": "string",
    "avatar_url": "string",
    "personal_email": "string",
    "personal_phone_number": "string",
    "work_email": "string",
    "work_phone_number": "string",
    "job_id": "string",
    "job_title": "string",
    "department_id": "string",
    "team_id": "string",
    "department": "string"
    // ... and 28 more fields
  }
}

### Sources
hris.json - Path: /unified/hris/employees/{id}
hris.json - Path: /unified/hris/employees
hris.json - Path: /unified/hris/employees/{id}/employments
hris.json - Path: /unified/hris/employees/{id}/employments/{subResourceId}

## C5 
### Answer
I don't have enough information in the StackOne documentation to answer this question. The context you provided describes the response body when retrieving a single employee (including its data object and fields), and when listing employee employments/employments, but it does not show the specific structure of the “list employees” response (e.g., whether it includes next, next_page, data, raw, etc.).

### Sources

hris.json - Path: /unified/hris/employees/{id}
hris.json - Path: /unified/hris/employees
hris.json - Path: /unified/hris/employees/{id}/employments/{subResourceId}
hris.json - Path: /unified/hris/employees/{id}/employments
hris.json - Path: /unified/hris/employments
