import useSWR from 'swr'
import { api } from '../../lib/api'
import InterviewPanel from '../components/interview/InterviewPanel'

export default function RegisterSimulator() {
  const { mutate: mutateRegistry } = useSWR('registry', () => api.getRegistry(), { revalidateOnFocus: false })

  return (
    <div className="page-shell">
      <InterviewPanel onRegistryUpdate={() => mutateRegistry()} />
    </div>
  )
}
